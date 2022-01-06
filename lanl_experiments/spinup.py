import math
import os
import pickle
import time

from sklearn.metrics import \
    roc_auc_score as auc_score, \
    f1_score, average_precision_score as ap_score
import torch 
import torch.distributed as dist 
import torch.distributed.rpc as rpc 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.optim import Adam, Adadelta

from loaders.tdata import TData
from models.euler_detector import DetectorEncoder, DetectorRecurrent 
from models.euler_predictor import PredictorEncoder, PredictorRecurrent
from models.utils import _remote_method_async, _remote_method
from utils import get_score, get_optimal_cutoff

DDP_PORT = '22032'
RPC_PORT = '22204'

DEFAULT_TR = {
    'lr': 0.01,
    'epochs': 3,
    'min': 1,
    'patience': 5,
    'nratio': 1,
    'val_nratio': 1,
}

# Defaults
WORKER_ARGS = [32,32]
RNN_ARGS = [32,32,16,1]

WORKERS=4
W_THREADS=1
M_THREADS=2

TMP_FILE = 'tmp.dat'
SCORE_FILE = 'scores.txt'

# Callable that returns TData object
# method signature must match
# workers: int, start=int, end=int, delta=int, is_test=bool 
LOAD_FN = None

torch.set_num_threads(1)

'''
Constructs params for data loaders
'''
def get_work_units(num_workers, start, end, delta, isTe):
    slices_needed = math.ceil((end-start) / delta)

    # Puts minimum tasks on each worker with some remainder
    per_worker = [slices_needed // num_workers] * num_workers 

    remainder = slices_needed % num_workers 
    if remainder:
        # Put remaining tasks on last workers since it's likely the 
        # final timeslice is stopped hallambda_paramay (ie it's less than a delta
        # so giving it extra timesteps is more likely okay)
        for i in range(num_workers, num_workers-remainder, -1):
            per_worker[i-1]+=1 

    # Only uncomment when running late at night
    load_threads = W_THREADS*2 if isTe else W_THREADS
    #load_threads = W_THREADS

    # Make sure workers are collectively using at least 8 threads
    # since loading the data takes forever otherwise
    min_threads = min(8, load_threads*num_workers)
    t_per_worker = max(1, min_threads//num_workers)

    print("Tasks: %s" % str(per_worker))
    kwargs = []
    prev = start
    
    for i in range(num_workers):
            end_t = min(prev + delta*per_worker[i], end)
            kwargs.append({
                'start': prev,
                'end': end_t,
                'delta': delta, 
                'is_test': isTe,
                'jobs': t_per_worker
            })
            prev = end_t

    return kwargs
    

def init_workers(num_workers, start, end, delta, isTe, worker_constructor, worker_args):
    kwargs = get_work_units(num_workers, start, end, delta, isTe)

    rrefs = []
    for i in range(len(kwargs)):
        rrefs.append(
            rpc.remote(
                'worker'+str(i),
                worker_constructor,
                args=(LOAD_FN, kwargs[i], *worker_args),
                kwargs={'head': i==0}
            )
        )

    return rrefs

def init_empty_workers(num_workers, worker_constructor, worker_args):
    empty = {'jobs': 0, 'start': None, 'end': None}
    
    rrefs = [
        rpc.remote(
            'worker'+str(i),
            worker_constructor,
            args=(LOAD_FN, empty, *worker_args),
            kwargs={'head': i==0}
        )
        for i in range(num_workers)
    ]

    return rrefs

def init_procs(rank, world_size, rnn_constructor, rnn_args, worker_constructor, worker_args, 
                times, just_test, lambda_param, impl, load_fn, tr_args):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = DDP_PORT

    # RPC info
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method='tcp://localhost:' + RPC_PORT

    # This is a lot easier than actually changing it in all the methods
    # at this point
    global LOAD_FN
    LOAD_FN = load_fn

    # Master (RNN module)
    if rank == world_size-1:
        torch.set_num_threads(M_THREADS)
        rpc.init_rpc(
            'master', rank=rank, 
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )


        # Evaluating a pre-trained model, so no need to train 
        if just_test:
            rrefs = init_empty_workers(
                world_size-1, 
                worker_constructor, 
                worker_args
            )

            rnn = rnn_constructor(*rnn_args)
            model = DetectorRecurrent(rnn, rrefs) if impl=='DETECT'\
                else PredictorRecurrent(rnn, rrefs)

            states = pickle.load(open('model_save.pkl', 'rb'))
            model.load_states(*states['states'])
            h0 = states['h0']
            tpe = 0
            tr_time = 0


        # Building and training a fresh model
        else:
            rrefs = init_workers(
                world_size-1, 
                times['tr_start'], times['tr_end'], times['delta'], False,
                worker_constructor, worker_args
            )

            tmp = time.time()
            model, h0, tpe = train(rrefs, tr_args, rnn_constructor, rnn_args, impl)
            tr_time = time.time() - tmp
        
        h0, zs = get_cutoff(model, h0, times, tr_args, lambda_param)
        stats = []

        for te_start,te_end in times['te_times']:
            test_times = {
                'te_start': te_start,
                'te_end': te_end,
                'delta': times['delta']
            }
            st = test(model, h0, test_times, rrefs)
            for s in st:
                s['TPE'] = tpe
                s['tr_time'] = tr_time

            stats += st

    # Slaves
    else:
        torch.set_num_threads(W_THREADS)
        
        # Slaves are their own process group. This allows
        # DDP to work between these processes
        dist.init_process_group(
            'gloo', rank=rank, 
            world_size=world_size-1
        )

        rpc.init_rpc(
            'worker'+str(rank),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

    # Block until all procs complete
    rpc.shutdown()

    # Write output to a tmp file to get it back to the parent process
    if rank == world_size-1:
        pickle.dump(stats, open(TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)


def train(rrefs, kwargs, rnn_constructor, rnn_args, impl):
    rnn = rnn_constructor(*rnn_args)
    model = DetectorRecurrent(rnn, rrefs) if impl=='DETECT' \
        else PredictorRecurrent(rnn, rrefs)

    opt = DistributedOptimizer(
        Adam, model.parameter_rrefs(), lr=kwargs['lr']
    )

    times = []
    best = (model.save_states(), 0)
    no_progress = 0
    for e in range(kwargs['epochs']):
        # Get loss and send backward
        model.train()
        with dist_autograd.context() as context_id:
            print("forward")
            st = time.time()
            zs = model.forward(TData.TRAIN)
            loss = model.loss_fn(zs, TData.TRAIN, nratio=kwargs['nratio'])

            print("backward")
            dist_autograd.backward(context_id, loss)
            
            print("step")
            opt.step(context_id)

            elapsed = time.time()-st 
            times.append(elapsed)
            l = torch.stack(loss).sum()
            print('[%d] Loss %0.4f  %0.2fs' % (e, l.item(), elapsed))

        # Get validation info to prevent overfitting
        model.eval()
        with torch.no_grad():
            zs = model.forward(TData.TRAIN, no_grad=True)
            p,n = model.score_edges(zs, TData.VAL)
            
            auc,ap = get_score(p,n)
            print("\tValidation: AP: %0.4f  AUC: %0.4f" % (ap, auc), end='')

            # Either incriment or update early stopping criteria
            tot = auc+ap
            if tot > best[1]:
                print('*\n')
                best = (model.save_states(), tot)
                no_progress = 0
            else:
                print('\n')
                if e >= kwargs['min']:
                    no_progress += 1 

            if no_progress == kwargs['patience']:
                print("Early stopping!")
                break 

    model.load_states(*best[0])

    # Get the best possible h0 to eval with
    zs, h0 = model(TData.TEST, include_h=True)

    states = {'states': best[0], 'h0': h0}
    f = open('model_save.pkl', 'wb+')
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

    tpe = sum(times)/len(times)
    print("Exiting train loop")
    print("Avg TPE: %0.4fs" % tpe)
    
    return model, h0, tpe


'''
Given a trained model, generate the optimal cutoff point using
the validation data
'''
def get_cutoff(model, h0, times, kwargs, lambda_param):
    # Weirdly, calling the parent class' method doesn't work
    # whatever. This is a hacky solution, but it works
    Encoder = DetectorEncoder if isinstance(model, DetectorRecurrent) \
        else PredictorEncoder

    # First load validation data onto one of the GCNs
    _remote_method(
        Encoder.load_new_data,
        model.gcns[0],
        LOAD_FN,
        {
            'start': times['val_start'],
            'end': times['val_end'],
            'delta': times['delta'],
            'jobs': 2,
            'is_test': False
        }
    )

    # Then generate GCN embeds
    model.eval()
    zs = _remote_method(
        Encoder.forward,
        model.gcns[0], 
        TData.ALL,
        True
    )

    # Finally, generate actual embeds
    with torch.no_grad():
        zs, h0 = model.rnn(zs, h0, include_h=True)

    # Then score them
    p,n = _remote_method(
        Encoder.score_edges, 
        model.gcns[0],
        zs, TData.ALL,
        kwargs['val_nratio']
    )

    # Finally, figure out the optimal cutoff score
    model.cutoff = get_optimal_cutoff(p,n,fw=lambda_param)

    print()
    return h0, zs[-1]

def test(model, h0, times, rrefs):
    # For whatever reason, it doesn't know what to do if you call
    # the parent object's methods. Kind of defeats the purpose of 
    # using OOP at all IMO, but whatever
    Encoder = DetectorEncoder if isinstance(model, DetectorRecurrent) \
        else PredictorEncoder

    # Load train data into workers
    ld_args = get_work_units(
        len(rrefs), 
        times['te_start'], 
        times['te_end'],
        times['delta'], 
        True
    )

    print("Loading test data")
    
    # Make sure there's enough data for each worker to do something
    dont_use = 0
    for ld in ld_args:    
        if ld['start'] == ld['end']:
            dont_use += 1
        else:
            break

    # If we have more workers than work. Tell master not to use them
    futs = [
        _remote_method_async(
            Encoder.load_new_data,
            rrefs[i], 
            LOAD_FN, 
            ld_args[i+dont_use]
        ) for i in range(len(rrefs)-dont_use)
    ]
    model.num_workers = len(futs)

    # Wait until all workers have finished
    [f.wait() for f in futs]
    stats = []

    with torch.no_grad():
        model.eval()
        s = time.time()
        zs = model.forward(TData.TEST, h0=h0, no_grad=True)
        ctime = time.time()-s

    # Scores all edges and matches them with name/timestamp
    print("Scoring")
    scores, labels, weights = model.score_all(zs)
    stats.append(
            score_stats(
            model.__class__.__name__, 
            scores, labels, weights, model.cutoff, ctime
        )       
    )
    
    # Then reset model to having all workers for future tests
    model.num_workers = len(rrefs)
    return stats
    

def score_stats(title, scores, labels, weights, cutoff, ctime):
    # Cat scores from timesteps together bc separation 
    # is no longer necessary 
    scores = torch.cat(scores, dim=0).detach()
    labels = torch.cat(labels, dim=0).clamp(max=1)
    weights = torch.cat(weights, dim=0)

    # Classify using cutoff from earlier
    classified = torch.zeros(labels.size())
    classified[scores <= cutoff] = 1

    # Calculate TPR
    p = classified[labels==1]
    tpr = p.mean()
    tp = p.sum()
    del p

    # Calculate FPR
    f = classified[labels==0]
    fp = f.sum()
    fpr = f.mean()
    del f 
    
    # Because a low score correlates to a 1 lable, sub from 1 to get
    # accurate AUC/AP scores
    scores = 1-scores

    # Get metrics
    auc = auc_score(labels, scores)
    ap = ap_score(labels, scores)
    f1 = f1_score(labels, classified)

    print(title)
    print("Learned Cutoff %0.4f" % cutoff)
    print("TPR: %0.4f, FPR: %0.4f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f" % f1)
    print("AUC: %0.4f  AP: %0.4f\n" % (auc,ap))

    return {
        'Model': title,
        'TPR':tpr.item(), 
        'FPR':fpr.item(), 
        'TP':tp.item(), 
        'FP':fp.item(), 
        'F1':f1, 
        'AUC':auc, 
        'AP': ap,
        'FwdTime':ctime
    }

def run_all(workers, rnn_constructor, rnn_args, worker_constructor, 
            worker_args, delta, just_test, lambda_param, impl, load_fn, 
            tr_start, tr_end, val_times, te_times, tr_args):
    '''
    Starts up proceses, trains validates and tests the model given 
    the inputs 

        workers : int 
            how many worker processes to use
        rnn_constructor : callable -> RNN 
            constructor for RNN model
        rnn_args : list 
            arguments for detector rnn model
        worker_constructor : callable -> Euler_Encoder_Unit 
            constructs an Euler_Encoder wrapped RRef to worker
        worker_args : list 
            non-file loading related worker arguments
        delta : int 
            size of time window to partition graphs
        just_test : boolean 
            Loads pre-trained model from disk and evaluates it
        lambda_param : float
            How much weight to give low FPR when deciding a cutoff;
            defaults to 0.6
        impl : str in ['DETECT', 'PREDICT']
            Class implimenting Euler_Interface
        load_fn : callable -> TGraph
            Function to load a set of snapshots into workers
        tr_start : int
            Timestep the training set starts at
        tr_end : int 
            Timestep the training set ends at
        te_end : int 
            Timestep the test set ends at. By default, loads the full LANL dataset
        tr_args : dict
            Hyperparameters for training. E.g. epochs, patience, etc. 
        '''
    
    # Need at least 2 deltas; default to 5% of tr data if that's enough
    if val_times is None:
        val = max((tr_end - tr_start) // 20, delta*2)
        val_start = tr_end-val
        val_end = tr_end
        tr_end = val_start
    else:
        val_start = val_times[0]
        val_end = val_times[1]

    # Make sure each worker has some data on it
    max_workers = int((tr_end-tr_start) // delta)
    workers = max(min(max_workers, workers), 1)

    times = {
        'tr_start': tr_start,
        'tr_end': tr_end,
        'val_start': val_start,
        'val_end': val_end,
        'te_times': te_times,
        'delta': delta
    }

    print(times)

    # Start workers
    world_size = workers+1
    mp.spawn(
        init_procs,
        args=(
            world_size, 
            rnn_constructor, 
            rnn_args, 
            worker_constructor, 
            worker_args,
            times,
            just_test,
            lambda_param,
            impl,
            load_fn,
            tr_args
        ),
        nprocs=world_size,
        join=True
    )

    # Retrieve stats, and cleanup temp file
    stats = pickle.load(open(TMP_FILE, 'rb'))
    #os.remove(TMP_FILE)

    print(stats)
    return stats

if __name__ == '__main__':
    print("Please run this file using run.py")
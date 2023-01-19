import json 
import os 

import pandas as pd 
import torch 
import torch.distributed as dist 
import torch.distributed.rpc as rpc 
import torch.multiprocessing as mp

from loaders.load_erdos import load_erdos
from loaders.tdata import TData
from timer_models import timed_gcn_rref, TimedRecurrent, TimedEmbedUnit
from euler.recurrent import GRU

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
WORKER_ARGS = [64,32]
RNN_ARGS = [32,32,16,1]

# Variables for testing
N_JOBS=50
WORKERS=8
W_THREADS=1
M_THREADS=1
N_SNAPSHOTS = W_THREADS

TMP_FILE = 'tmp.txt'

torch.set_num_threads(1)

def get_work_units(workers, snapshots):
    return [
        {
            'jobs': snapshots,
            'start': 0,
            'end': 0
        }
    ] * workers

def init_workers(num_workers, tot_snapshots):
    kwargs = get_work_units(num_workers, tot_snapshots)

    rrefs = []
    for i in range(len(kwargs)):
        rrefs.append(
            rpc.remote(
                'worker'+str(i),
                timed_gcn_rref,
                args=(load_erdos, kwargs[i], *WORKER_ARGS),
                kwargs={'head': i==0}
            )
        )

    return rrefs

def time_test(model):
    _, stats = model.forward(TData.TRAIN, batched=False)
    print(json.dumps(stats, indent=1))
    
    stats.pop('full'); stats.pop('if_serial')
    return {
        'enc': sum(stats['enc']),
        'rnn': sum(stats['rnn']),
        'waiting': sum(stats['waiting']),
        'comm_time': stats['waiting'][0]-stats['enc'][0]
    }

def init_procs(rank, world_size, tot_jobs, w_threads):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = DDP_PORT

    # RPC info
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method='tcp://localhost:' + RPC_PORT

    # Master (RNN module)
    if rank == world_size-1:
        torch.set_num_threads(M_THREADS)

        rpc.init_rpc(
            'master', rank=rank, 
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

        stats = []
        for _ in range(1):
            rrefs = init_workers(
                world_size-1, 
                tot_jobs
            )

            rnn = GRU(*RNN_ARGS)
            model = TimedRecurrent(rnn, rrefs)
            stats.append(time_test(model))

            del rrefs, model, rnn 

        df = pd.DataFrame(stats)
        keys = ['enc', 'rnn', 'waiting', 'comm_time']
        mean = df.mean()
        sem = df.sem()
        out_str = '%d,%d,%d,' % (world_size-1, tot_jobs, w_threads)

        for k in keys:
            out_str += '%f,%f,' % (mean[k],sem[k])

        out_str = out_str[:-1]+'\n'

        with open('out.txt', 'a') as f:
            f.write(out_str)

    # Slaves
    else:
        torch.set_num_threads(w_threads)
        
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

def run_serial(tj, threads=8):
    torch.set_num_threads(threads)
    kw = {
        'jobs': tj,
        'start': 0,
        'end': 0
    }

    # TODO this doesn't really work bc TimeEmbedUnit inherits from DDP
    # so rpc must be initialized even though we arent using it. Not worth
    # the effort of writing a new class, or initializing it for no reason imo
    stats = []
    for _ in range(10):
        model = TimedEmbedUnit(load_erdos, kw, *WORKER_ARGS)
        _,t = model.forward(TData.TRAIN)
        stats.append(t)

        del model 

    with open('out.txt', 'a') as f:
        f.write('%d,%f\n' % (tj,sum(stats)/10))
    

def run_all(workers, multiplier, w_threads):
    # Start workers
    world_size = workers+1
    mp.spawn(
        init_procs,
        args=(world_size, multiplier, w_threads),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    #for tot_jobs in [1024, 512, 256, 128, 64, 32, 16]:
    #    for ws in [16, 14, 12, 10, 8, 6, 4, 2, 1]:
    #        run_all(1, tot_jobs//ws, ws)
    #
    for tj in [1536,2048,2560,3072,4096]:
        run_all(1, tj, 8) # 8 threads is about optimal
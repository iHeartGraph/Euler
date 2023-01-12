import time 
import json 
import os

import torch 
import torch.distributed as dist 
import torch.distributed.rpc as rpc 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.optim import Adam

from loaders.load_erdos import load_erdos
from loaders.tdata import TData
from timer_models import timed_gcn_rref, TimedRecurrent
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
WORKER_ARGS = [32,32]
RNN_ARGS = [32,32,16,1]

# Variables for testing
WORKERS=1
W_THREADS=1
M_THREADS=1
N_SNAPSHOTS = 2

torch.set_num_threads(1)

def get_work_units(workers, snapshots):
    return [2]*workers # TODO 

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
    zs, stats = model.forward(TData.tr)
    print(json.dumps(stats, indent=1))

def init_procs(rank, world_size):
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

        rrefs = init_workers(
            world_size-1, 
            N_SNAPSHOTS
        )

        rnn = GRU(*RNN_ARGS)
        model = TimedRecurrent(rnn, rrefs)

        time_test(model)

    # Slaves
    else:
        time.sleep(5)
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



def run_all(workers):
    # Start workers
    world_size = workers+1
    mp.spawn(
        init_procs,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    run_all(WORKERS)
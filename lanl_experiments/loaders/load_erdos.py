import math
import random 

import torch 

from .tdata import TData

def load_erdos(jobs, n=1000, p=0.2, edge_high=20000, edge_low=10000, **kws):
    src,dst = [],[]
    eis = []
    masks = []

    v = 1
    w = -1
    lp = math.log(p)

    edges_per = [0,random.randint(edge_low, edge_high)]
    
    while True:
        lr = math.log(1.0 - random.random())
        w = w + 1 + int(lr / lp)
        while w >= v and v < n:
            w = w - v
            v = v + 1
        if v < n:
            src.append(w)
            dst.append(v)

            if len(src) == edges_per[-1]:
                # Stop when we have enough snapshots
                if len(edges_per) == jobs+1:
                    break 
            
                # Otherwise get more
                ep = edges_per[-1]
                edges_per.append(ep+random.randint(edge_low, edge_high))

        # Instead of terminating, keep going until 
        # enough edges generated
        v %= n 

    # Then shuffle them around
    ei = torch.tensor([src,dst])
    ei = ei[:, torch.randperm(ei.size(1))]
    mask = torch.rand(ei.size(1)) > 0.05

    # Then split into snapshots
    for i in range(len(edges_per)-1):
        st = edges_per[i]; end = edges_per[i+1]
        eis.append(ei[:, st:end])
        masks.append(mask[st:end])

    x = torch.eye(n)
    return TData(
        eis, x, None, masks
    )
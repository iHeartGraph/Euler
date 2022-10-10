from copy import deepcopy
import os 
import pickle 
from joblib import Parallel, delayed

import torch 
from torch_geometric.data import Data 
from tqdm import tqdm 

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized

DATE_OF_EVIL_LANL = 150885
FILE_DELTA = 10000

# Input where LANL data cleaned with .clean_lanl.py is stored
LANL_FOLDER = None
assert LANL_FOLDER, 'Please fill in the LANL_FOLDER variable:\n line 17 /lanl_experiments/loaders/load_lanl.py'

COMP = 0
USR = 1
SPEC = 2
X_DIM = 17688

TIMES = {
    20      : 228642, # First 20 anoms
    100     : 740104, # First 100 anoms
    500     : 1089597, # First 500 anoms
    'all'  : 5011199  # Full
}

torch.set_num_threads(1)

def empty_lanl():
    return make_data_obj([],None,None)

def load_lanl_dist(workers, start=0, end=635015, delta=8640, is_test=False, ew_fn=std_edge_w):
    if start == None or end == None:
        return empty_lanl()

    num_slices = ((end - start) // delta)
    remainder = (end-start) % delta
    num_slices = num_slices + 1 if remainder else num_slices
    workers = min(num_slices, workers)

    # Can't distribute the job if not enough workers
    if workers <= 1:
        return load_partial_lanl(start, end, delta, is_test, ew_fn)

    per_worker = [num_slices // workers] * workers 
    remainder = num_slices % workers

    # Give everyone a balanced number of tasks 
    # put remainders on last machines as last task 
    # is probably smaller than a full delta
    if remainder:
        for i in range(workers, workers-remainder, -1):
            per_worker[i-1] += 1

    kwargs = []
    prev = start 
    for i in range(workers):
        end_t = prev + delta*per_worker[i]
        kwargs.append({
            'start': prev, 
            'end': min(end_t-1, end),
            'delta': delta,
            'is_test': is_test,
            'ew_fn': ew_fn
        })
        prev = end_t
    
    # Now start the jobs in parallel 
    datas = Parallel(n_jobs=workers, prefer='processes')(
        delayed(load_partial_lanl_job)(i, kwargs[i]) for i in range(workers)
    )

    # Helper method to concatonate one field from all of the datas
    data_reduce = lambda x : sum([getattr(datas[i], x) for i in range(workers)], [])

    # Just join all the lists from all the data objects
    print("Joining Data objects")
    x = datas[0].xs
    eis = data_reduce('eis')
    masks = data_reduce('masks')
    ews = data_reduce('ews')
    node_map = datas[0].node_map

    if is_test:
        ys = data_reduce('ys')
        cnt = data_reduce('cnt')
    else:
        ys = None
        cnt = None

    # After everything is combined, wrap it in a fancy new object, and you're
    # on your way to coolsville flats
    print("Done")
    return TData(
        eis, x, ys, masks, ews=ews, node_map=node_map, cnt=cnt
    )
 

# wrapper bc its annoying to send kwargs with Parallel
def load_partial_lanl_job(pid, args):
    data = load_partial_lanl(**args)
    return data


def make_data_obj(eis, ys, ew_fn, ews=None, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

    cl_cnt = len(nm)
    x = torch.eye(cl_cnt+1)
    
    # Build time-partitioned edge lists
    eis_t = []
    masks = []

    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)

        # This is training data if no ys present
        if isinstance(ys, None.__class__):
            masks.append(edge_tv_split(ei)[0])

    # Balance the edge weights if they exist
    if not isinstance(ews, None.__class__):
        cnt = deepcopy(ews)
        ews = ew_fn(ews)
    else:
        cnt = None

    # Finally, return Data object
    return TData(
        eis_t, x, ys, masks, ews=ews, cnt=cnt, node_map=nm
    )

'''
Equivilant to load_cyber.load_lanl but uses the sliced LANL files 
for faster scanning to the correct lines
'''
def load_partial_lanl(start=140000, end=156659, delta=8640, is_test=False, ew_fn=standardized):
    cur_slice = int(start - (start % FILE_DELTA))
    start_f = str(cur_slice) + '.txt'
    in_f = open(LANL_FOLDER + start_f, 'r')

    edges = []
    ews = []
    edges_t = {}
    ys = []

    # Predefined for easier loading so everyone agrees on NIDs
    node_map = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

    # Helper functions (trims the trailing \n)
    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[3][:-1]))

    # For now, just keeps one copy of each edge. Could be
    # modified in the future to add edge weight or something
    # but for now, edges map to their anomaly value (1 == anom, else 0)
    def add_edge(et, is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = (max(is_anom, val[0]), val[1]+1)
        else:
            edges_t[et] = (is_anom, 1)


    scan_prog = tqdm(desc='Finding start', total=start-cur_slice-1)
    prog = tqdm(desc='Seconds read', total=end-start-1)

    anom_marked = False
    keep_reading = True
    next_split = start+delta 

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime 
    while keep_reading:
        while line:
            l = line.split(',')
            
            # Scan to the correct part of the file
            ts = int(l[0])
            if ts < start:
                line = in_f.readline()
                scan_prog.update(ts-old_ts)
                old_ts = ts 
                curtime = ts 
                continue
            
            ts, src, dst, label = fmt_line(l)
            et = (src,dst)

            # Not totally necessary but I like the loading bar
            prog.update(ts-old_ts)
            old_ts = ts

            # Split edge list if delta is hit 
            if ts >= next_split:
                if len(edges_t):
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    y,ew = list(zip(*edges_t.values()))
                    ews.append(torch.tensor(ew))

                    if is_test:
                        ys.append(torch.tensor(y))

                    edges_t = {}

                # If the list was empty, just keep going if you can
                curtime = next_split 
                next_split += delta

                # Break out of loop after saving if hit final timestep
                if ts >= end:
                    keep_reading = False 
                    break 

            # Skip self-loops
            if et[0] == et[1]:
                line = in_f.readline()
                continue

            add_edge(et, is_anom=label)
            line = in_f.readline()

        in_f.close() 
        cur_slice += FILE_DELTA 

        if os.path.exists(LANL_FOLDER + str(cur_slice) + '.txt'):
            in_f = open(LANL_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
        else:
            keep_reading=False
            break

    ys = ys if is_test else None

    scan_prog.close()
    prog.close()

    return make_data_obj(
        edges, ys, ew_fn,
        ews=ews, node_map=node_map
    )


if __name__ == '__main__':
    data = load_lanl_dist(2, start=0, end=21600, delta=21600)
    print(data)
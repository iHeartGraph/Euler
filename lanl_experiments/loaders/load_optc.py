from copy import deepcopy
from joblib import Parallel, delayed
import math
import os.path

import pickle
import torch 
from tqdm import tqdm 

from .tdata import TData
from .load_utils import edge_tv_split, standardized, std_edge_w

# Important files
# Unfortunately, the code that built these files was on a disk that 
# died and wasn't backed up on the repo. But to generate them, just
# parse the OpTC data in order, and pull out all FLOW-START records
DATA_FOLDER = '/home/ead/datasets/OpTC/flow_split/'

# For ease of reading. The headers for the original CSV
TS = 0
SRC = 1
DST = 2 
Y = 3
SNAPSHOT = 4

# How many snapshots per split file
F_DELTA = 100

# For convinience pulled out of nodemap file
NUM_NODES = 1114
NUM_ANOMS = 21641

# Times expressed in minutes since start of data 
TIMES = {
    'first_anom': 9559, # About 6.5 days in
    'last_anom': 12300, # About 8.5 days in
    'all': 12635,       # 8.77 days
    'tr_end': 3101,      # Run until the first major data gap
    'val_start': 3531,   # Use datagap as val data?
    'val_end': 4097
}

# Test regions
DAY1 = (8845, 9876)
DAY2 = (10831, 11756)
DAY3 = (11756, 12635)
ALL = (DAY1[0], DAY3[1])

# Spans of time with no data at all
DEAD_ZONES = [
    (4097, 8845),
    (9876, 10831)
]


def load_optc_dist(workers, start=0, end=None, delta=30, is_test=False, weight_fn=std_edge_w):
    # Return empty data object if start is None
    if start is None:
        return TData([],torch.eye(NUM_NODES),None,[])

    # Just run normally if only 1 worker
    if workers <= 1: 
        eis, ews, ys = load_optc(
            start=start, 
            end=end,
            delta=delta, 
            is_te=is_test
        )

        return build_data_obj(eis, ews, ys, weight_fn)

    # Otherwise, split the work as evenly as possible across 
    # worker threads..
    work_units = math.ceil((end-start)/delta)
    workers = min(workers, work_units)
    jobs_per_worker = [work_units // workers] * workers 

    # Add remaining jobs to latter workers 
    remainder = work_units % workers 
    for w in range(remainder):
        jobs_per_worker[workers-1-w] += 1

    # Convert from jobs to start/end
    kwargs = []
    for w in range(workers):
        kwargs.append(
            {
                'start': start, 
                'end': min(
                    start + delta*jobs_per_worker[w],
                    end
                ), 
                'delta': delta,
                'is_te': is_test
            }
        )

        start += delta*jobs_per_worker[w]

    # Then execute all jobs
    snapshots = Parallel(n_jobs=workers, prefer='processes')(
        delayed(optc_job)(kwargs[i], weight_fn) for i in range(workers)
    )

    # Merge all lists into one list of snapshot datas
    eis, ews, ys, masks, cnt = [], [], [], [], []
    for s in snapshots:
        eis.append(s['eis'])
        ews.append(s['ews'])
        ys.append(s['ys'])
        masks.append(s['masks'])
        cnt.append(s['cnt'])

    # Then return the TData object
    return TData(
        sum(eis, []),
        torch.eye(NUM_NODES), 
        sum(ys, []) if is_test else None,
        sum(masks, []),
        ews=sum(ews, []),
        cnt=sum(cnt, []) if is_test else None
    )


def optc_job(kwargs, weight_fn):
    eis, ews, ys = load_optc(**kwargs)
    
    # Before compressing into weighted edges, save
    # raw count of edges for later testing so repeated edges
    # are scored each time they appear
    if ys is not None:
        cnt = deepcopy(ews)
    else:
        cnt = None 

    ews = weight_fn(ews)
    masks = [edge_tv_split(ei)[0] for ei in eis]

    return {
        'eis': eis, 
        'ews': ews, 
        'ys': ys, 
        'masks': masks,
        'cnt': cnt
    }


def build_data_obj(eis, ews, ys, weight_fn):
    if ys is not None:
        cnt = deepcopy(ews)
    else:
        cnt = None

    ews = weight_fn(ews)
    
    # Only build masks if training
    masks = [edge_tv_split(ei)[0] for ei in eis] \
        if ys is None else []

    return TData(
        eis, 
        torch.eye(NUM_NODES),
        ys, 
        masks, 
        ews=ews,
        cnt=cnt
    )

def load_optc(start=0, end=None, delta=30, is_te=False):
    '''refactoring below'''
    format_f = lambda x : DATA_FOLDER + str(x) + '.csv'

    if end is None: 
        end = TIMES['all']

    snapshots = []
    tot_snapshots = math.ceil((end-start) / delta)
    for i in range(tot_snapshots):
        snapshots.append(
            [
                format_f(j) for j in range(
                    i*delta + start, 
                    min([(i+1)*delta + start, end])
                )
            ]
        )

    eis=[]; ews=[]; ys=[]
    for i,snapshot in enumerate(snapshots):
        ei,ew,y = build_one_snapshot(snapshot, is_te, i, len(snapshots))
        if type(ei) == torch.Tensor:
            eis.append(ei)
            ews.append(ew)
            ys.append(y)

    return eis, ews, ys if is_te else None

def build_one_snapshot(files, is_te, snap_idx, tot_snaps):
    def add_edge(ei, edge, label):
        # Edges are tuple of <count, isAnom>
        if edge in ei:
            cnt,anom = ei[edge]
            ei[edge] = (cnt+1, max(anom,label))
        else:
            ei[edge] = (1, label)

    ei = {}
    for file in tqdm(files, desc='Loading data (%d/%d)' % (snap_idx+1,tot_snaps)):
        if not os.path.isfile(file):
            continue 

        cur_f = open(file, 'r')
        line = cur_f.readline()
        while(line):
            line = line.split(',')
            
            src, dst = int(line[SRC]), int(line[DST])

            # Ignore self-loops
            if src != dst:
                label = int(line[Y])
                add_edge(ei, (src,dst), label)

            # Read in the next line, and incriment file if needed
            line = cur_f.readline()

        cur_f.close() 

    if ei:
        src,dst = list(zip(*ei.keys()))
        ews,labels = list(zip(*ei.values()))

        return (
            torch.tensor([src,dst]),
            torch.tensor(ews),
            torch.tensor(labels) if is_te else None
        )
    else:
        return [],[],[] if is_te else None

if __name__ == '__main__':
    load_optc(start=0, end=TIMES['tr_end'], delta=30)
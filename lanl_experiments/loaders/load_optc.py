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
IN_F = '/mnt/raid0_24TB/isaiah/data/optc/all_1min.csv'
DATA_FOLDER = '/mnt/raid0_24TB/isaiah/data/optc/split/'
NODE_MAP = '/mnt/raid0_24TB/isaiah/data/optc/nodemap.pkl'

# For ease of reading. The headers for the original CSV
TS = 1
SRC = 2
DST = 3 
Y = 4 
SNAPSHOT = 5

# How many snapshots per split file
F_DELTA = 100

# For convinience pulled out of nodemap file
NUM_NODES = 1114
NUM_ANOMS = 21641
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

# Only needs to be called once; splits the big file into lots of small 
# files holding only 30 mins of data a piece
def split():
    start = 0 
    end = F_DELTA
    fmt_file = lambda x : '%s/%d.csv' % (DATA_FOLDER, x)

    big_f = open(IN_F, 'r')
    small_f = open(fmt_file(start), 'w+')
    prog = tqdm(desc='Records parsed', total=40568698)

    all_nodes = {}
    cnt = [0]
    inverted = []

    def get_or_add(node, cnt): 
        if node not in all_nodes:
            all_nodes[node] = cnt[0] 
            inverted.append(node)
            cnt[0] += 1
        
        return all_nodes[node]

    # Skip header
    line = big_f.readline()
    line = big_f.readline()

    while(line):
        line = line.split(',')
        snp = int(float(line[SNAPSHOT]))

        if snp >= end:
            # Not all records are consecutive, as we've found
            # out.. there are big empty spaces in the data
            start = snp - (snp % F_DELTA)
            end = start + F_DELTA

            small_f.close()
            small_f = open(fmt_file(start), 'w+')

        # Change to unique node IDs
        line[SRC] = str(get_or_add(line[SRC], cnt))
        line[DST] = str(get_or_add(line[DST], cnt))
        line[SNAPSHOT] = str(snp)

        # Convert back to str and save
        line = ','.join(line) + '\n'
        small_f.write(line)
        line = big_f.readline() 
        prog.update()

    # Cleanup
    small_f.close() 
    big_f.close() 
    prog.close()

    with open(NODE_MAP, 'wb+') as f:
        pickle.dump(inverted, f)


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
    format_f = lambda x : DATA_FOLDER + str(x) + '.csv'

    # Find the file with the first record we're interested in
    # While loop protects edge case where starting on file that
    # DNE, and need to find the next available one
    cur_f_num = (start // F_DELTA) * F_DELTA
    fname = format_f(cur_f_num)
    skip_scan = False 

    while(not os.path.isfile(fname)):
        skip_scan = True
        cur_f_num += F_DELTA
        fname = format_f(cur_f_num)

        # Either skip over chunks of missing data, 
        # or break when we run out of files
        if cur_f_num >= end:
            return [], [], [] if is_te else None

    
    cur_f = open(format_f(cur_f_num), 'r')
    line = cur_f.readline()

    # If the start isn't at the start of the file
    if start % F_DELTA and not skip_scan: 
        prog = tqdm(desc='Finding start')

        while(line):
            prog.update()
            l = line.split(',')
            stamp = int(l[SNAPSHOT])
            
            if stamp >= start:
                line = l 
                break

            line = cur_f.readline()
        
        # If we didn't break from the loop, it means 
        # start was not in the file it should be in (i.e. data gap
        # s.t. the first log will be in the future)
        else:
            while(not os.path.isfile(fname)):
                cur_f_num += F_DELTA

                # Either skip over chunks of missing data, 
                # or break when we run out of files
                if cur_f_num >= end:
                    return [], [], [] if is_te else None

                fname = format_f(cur_f_num)

            cur_f = open(fname, 'r')
            line = cur_f.readline().split(',')

        prog.close()

    def add_edge(ei, edge, label):
        # Edges are tuple of <count, isAnom>
        if edge in ei:
            cnt,anom = ei[edge]
            ei[edge] = (cnt+1, max(anom,label))
        else:
            ei[edge] = (1, label)

    # Now we're sure `line` is at start, so we continue iterating until we hit
    # the end
    stamp = lambda x : int(x[SNAPSHOT])
    cur_delta = (stamp(line) // delta) * delta

    edges = []
    weights = []
    ys = []
    eof = False

    tot_deltas = math.ceil((end-start) / delta)
    prog = tqdm(desc='Loading data', total=tot_deltas)
    while(not eof and stamp(line) < end):
        ei = {}
        
        # Continue loading data until timewindow complete, or we reach the end
        while(not eof and stamp(line) < cur_delta + delta and stamp(line) < end):
            src, dst = int(line[SRC]), int(line[DST])

            # Ignore self-loops
            if src != dst:
                label = int(line[Y])
                add_edge(ei, (src,dst), label)

            # Read in the next line, and incriment file if needed
            line = cur_f.readline()
            if not line:
                cur_f.close()
                cur_f_num += F_DELTA
                fname = format_f(cur_f_num)

                # Make sure file exists
                while(not os.path.isfile(fname)):
                    cur_f_num += F_DELTA
                    fname = format_f(cur_f_num)

                    # Either skip over chunks of missing data, 
                    # or break when we run out of files
                    if cur_f_num >= TIMES['all']:
                        eof = True 
                        break 

                # Load in next file
                if not eof:
                    cur_f = open(fname, 'r')
                    line = cur_f.readline()

            line = line.split(',')

        # We have either filled out data for a full snapshot, or are out of data
        # either way, now it's time to convert to tensors and add to the 
        # edges/ys lists
        # Ei *should* only be empty in edge case where snapshots that DNE are in the
        # range asked for. 
        if len(ei):
            src,dst = list(zip(*ei.keys()))
            edges.append(torch.tensor([src,dst]))
            
            cnts,labels = list(zip(*ei.values()))
            weights.append(torch.tensor(cnts))
        
            # Don't waste memory saving a bunch of zeros if not the test set
            if is_te:
                ys.append(torch.tensor(labels))

        '''
        # If empty timestep, just return the ID matrix weights = 1
        else:
            src = list(range(NUM_NODES))
            edges.append(torch.tensor([src,src]))

            cnts = torch.full((NUM_NODES,), 1)
            weights.append(cnts)

            if is_te:
                labels = torch.zeros(cnts.size())
                ys.append(labels)
        '''

        cur_delta += delta
        prog.update()

    prog.close()
    return edges, weights, ys if is_te else None


if __name__ == '__main__':
    load_optc_dist(1, start=2322, end=3101, delta=6)
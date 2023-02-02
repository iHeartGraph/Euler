from collections import defaultdict
import gzip 
import glob 
import json 
from math import ceil
from multiprocessing import Manager, Pool, Lock
import os 

from dateutil import parser 
from joblib import Parallel, delayed
import pickle
from tqdm import tqdm 

RAW_OPTC = '/home/ead/datasets/OpTC/ecar/'
TRAIN = 'benign_gz/'
TEST = 'eval_gz/'
SPLIT = 'flow_split/'

# From checking the first/last line of each file
FIRST_TS = 1568676405.62800
LAST_TS  = 1569436694.309
F_DELTA = 100 # Num snapshots per file
SNAPSHOT_SIZE = 60 # seconds
MAX_BUFFER = 2**16 # Max lines workers store before flushing out

get_ts = lambda x : parser.parse(x).timestamp()
get_snapshot = lambda x : (x - FIRST_TS) // SNAPSHOT_SIZE 
host_to_idx = lambda x : int(x[9:13])

def label_line(src,dst,ts):
    return 0 # TODO 


def ignore(ip):
    if (
        ip.startswith('ff') or # Multicast
        ip.endswith('255') or  # Broadcast
        ip.endswith('1')       # Gateway
    ):
        return True 

    if ':' in ip:
        return False 

    # Check for multicast (224.0.0.0 - 239.255.255.255)
    first_octet = int(ip.split('.')[0])
    if first_octet >= 224 and first_octet <= 239:
        return True 

    return False

def build_maps(fold=TRAIN, out_f='flow_split/nmap.pkl'):
    files = glob.glob(RAW_OPTC+fold+'**/*.gz')
    print(len(files))
    
    maps = Parallel(n_jobs=64, prefer='processes')(
        delayed(build_map)(f,i,len(files)) for i,f in enumerate(files)
    )

    # Append to existing nodemap if possible
    # (e.g. running on test set after tr set)
    if os.path.exists(out_f):
        with open(out_f, 'rb') as f:
            node_map = pickle.load(f)
    else:
        node_map = maps.pop()

    [node_map.update(m) for m in maps]
    with open(out_f, 'wb+') as f:
        pickle.dump(node_map, f)


def split_all(nmap_f, fold=TRAIN):
    files = glob.glob(RAW_OPTC+fold+'**/*.gz')
    print(len(files))
    print(files[0])

    # Split into smaller files
    with open(nmap_f, 'rb') as f:
        node_map = pickle.load(f)

    with Manager() as manager:
        p = Pool(processes=16)
        lock = manager.Lock()

        # Start all jobs 
        tasks = [
            p.apply_async(copy_one, (f, node_map, lock, i, len(files)))
            for i,f in enumerate(files)
        ]
        print("Queued %d jobs" % len(tasks))

        for i,t in enumerate(tasks):
            t.wait()
            print("Finished (%d/%d)" % (i+1, len(tasks)))

        p.close()
        p.join()

def copy_one(in_f, node_map, lock, i, tot):
    in_f = gzip.open(in_f)

    line = in_f.readline()
    prog = tqdm(desc='%d/%d' % (i+1, tot))
    
    buffer = None 
    io_buffer = defaultdict(str)
    buffer_contents = 0 
    while line:
        datum = json.loads(line)
    
        if datum['action'] == 'START' and datum['object'] == 'FLOW':
            props = datum['properties']
            src_ip = props['src_ip']
            dst_ip = props['dest_ip']
            host = datum['hostname']

            if props['direction'] == 'inbound':
                src = node_map.get(src_ip)
                dst = host 
            elif props['direction'] == 'outbound':
                src = host
                dst = node_map.get(dst_ip)

            # Only log if we can attribute src and dst hosts
            if src and dst:
                # Avoid repeats from repeated requests
                if (src,dst) == buffer:
                    line = in_f.readline()
                    prog.update() 
                    continue
                else:
                    buffer = (src,dst)

                ts = get_ts(datum['timestamp'])
                y = label_line(src,dst,ts)

                src = host_to_idx(src)
                dst = host_to_idx(dst)
                snapshot = get_snapshot(ts)

                io_buffer[snapshot] += '%f,%d,%d,%d\n' % (ts,src,dst,y)
                buffer_contents += 1
            
                # Try to minimize syncrhonization 
                if buffer_contents > MAX_BUFFER:
                    for f_num,out_str in io_buffer.items():
                        fname = SPLIT+str(int(f_num)) + '.csv'
                        
                        with lock:
                            out_f = open(fname, 'a+')
                            out_f.write(out_str)
                            out_f.close() 

                    # Empty buffer
                    io_buffer = defaultdict(str)
                    buffer_contents = 0

        line = in_f.readline()
        prog.update() 

    # Write before returning regardless of buffer len
    for f_num,out_str in io_buffer.items():
        fname = SPLIT+str(int(f_num)) + '.csv'
        
        with lock:
            out_f = open(fname, 'a+')
            out_f.write(out_str)
            out_f.close() 

    prog.close()


def build_map(in_f, i, tot):
    ip_map = dict()
    f = gzip.open(in_f)

    line = f.readline()
    prog = tqdm(desc='%d/%d' % (i+1, tot))
    
    cnt = 0
    while line: # and cnt < 100000:
        datum = json.loads(line)
    
        if datum['object'] == 'FLOW':
            props = datum['properties']
            src_ip = props['src_ip']
            
            if not (dst_ip := props.get('dest_ip')):
                # I have no clue what causes this. Seems like just flow-open events?
                # print(json.dumps(datum,indent=1))
                line = f.readline()
                prog.update()
                continue 


            ip = dst_ip if props['direction'] == 'inbound' else src_ip
            if ip not in ip_map:
                if not ignore(ip): # and ip.startswith('142')  
                    host = datum['hostname']
                else:
                    host = None 
            
                # Add to mapping
                ip_map[ip] = host 
            cnt += 1

        line = f.readline()
        prog.update() 

    prog.close()
    return ip_map

if __name__ == '__main__':
    #build_maps()
    #build_maps(fold=TEST)
    split_all('flow_split/nmap.pkl')

    '''
    # Testing
    with open('flow_split/nmap.pkl', 'rb') as f:
        node_map = pickle.load(f)

    copy_one(
        '/home/ead/datasets/OpTC/ecar/benign_gz/17-18Sep19/AIA-51-75.ecar-last.json.gz',
        node_map, Lock(), 0,1
    )
    '''
from collections import defaultdict
import gzip 
import glob 
import json 
import os 

from dateutil import parser 
from joblib import Parallel, delayed
import pickle
from tqdm import tqdm 

RAW_OPTC = '/home/ead/datasets/OpTC/ecar/'
TRAIN = 'benign_gz/'
TEST = 'eval_gz/'
SPLIT = 'flow_split/'
PARSED = 'flow_starts'

# From checking the first line of each file
FIRST_TS = 1568676710.696
get_ts = lambda x : parser.parse(x[:-1]).timestamp()

def label_line():
    pass # TODO 

def split(fname, i,tot):
    '''
    Takes lines from file formatted as 
        src_ip, dst_ip, ts
    Outputs lines in file (minutes since start//100).csv
        ts, src_uuid, dst_uuid, minute, label
    
    Since all files span huge timespans it's unlikely that both 
    would be writing to the same file at once... but not impossible.
    May need to impliment locking or something TODO 
    '''
    in_f = open(fname, 'r')
    line = in_f.readline()
    prog = tqdm(desc='%d/%d' % (i+1, tot))

    while line:
        src,dst,ts = line.split(',')

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

def parse_raw(fold=TRAIN, out_f='nmap.pkl'):
    files = glob.glob(RAW_OPTC+fold+'**/*.gz')
    print(len(files))
    
    maps = Parallel(n_jobs=64, prefer='processes')(
        delayed(copy_one)(f,i,len(files)) for i,f in enumerate(files)
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

def copy_one(in_f, i, tot):
    ip_map = dict()
    f = gzip.open(in_f)
    spl = in_f.split('/')
    stem = spl[-1]
    fold = spl[-3][:-3]

    out_f = '/'.join([PARSED,fold,stem])[:-7] + 'csv'
    out_f = open(out_f, 'w+')

    line = f.readline()

    prog = tqdm(desc='%d/%d' % (i+1, tot))
    
    cnt = 0
    buffer = None 
    while line:
        datum = json.loads(line)
    
        if datum['action'] == 'START' and datum['object'] == 'FLOW':
            props = datum['properties']
            src_ip = props['src_ip']
            dst_ip = props['src_ip']

            # Avoid repeats from repeated requests
            if (src_ip,dst_ip) == buffer:
                line = f.readline()
                prog.update() 
                continue
            else:
                buffer = (src_ip,dst_ip)

            out_f.write( ','.join([src_ip,dst_ip,datum['timestamp']]) + '\n')

            ip = dst_ip if props['direction'] == 'inbound' else src_ip

            if ip not in ip_map:      
                if not ignore(ip):
                    host = datum['hostname']
                else:
                    host = None 

                # Add to mapping
                ip_map[ip] = host 
                cnt += 1

        line = f.readline()
        prog.update() 

    prog.close()
    out_f.close()

    return ip_map



if __name__ == '__main__':
    parse_raw(fold=TEST)
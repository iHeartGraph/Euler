from collections import defaultdict
import gzip 
import glob 
import json 

from joblib import Parallel, delayed
import pickle
from tqdm import tqdm 

RAW_OPTC = '/home/ead/datasets/OpTC/ecar/'
TRAIN = 'benign_gz/'
TEST = 'evaluation_gz/'
PARSED = 'flow_starts'

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
    node_map = maps.pop()
    [node_map.update(m) for m in maps]

    with open(out_f, 'wb+') as f:
        pickle.dump(out_f, f)

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
    parse_raw()
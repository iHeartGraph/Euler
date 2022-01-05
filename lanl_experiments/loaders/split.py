import os 
import pickle 
from tqdm import tqdm

# LANL is so huge it's prohibitively expensive to scan it for 
# edges of later time steps. To remedy this (and make it easier
# for the distro models to load data later on) I have split it into
# files containing 10,000 seconds each 

# Please obtain the LANL data set from:
# https://csr.lanl.gov/data/cyber1/

RED = '' # Location of redteam.txt
SRC = '' # Location of auth.txt
DST = '' # Directory to save output files to

assert RED and SRC and DST, 'Please download the LANL data set, and mark in the code where it is:\nLines 13-15 of cyber/loaders/split.py'

DELTA = 10000
DAY = 60**2 * 24

def mark_anoms():
    '''
    Parses the redteam file and creates a small dict of 
    nodes involved with anomalous edges, and when they occur
    '''
    with open(RED, 'r') as f:
        red_events = f.read().split()

    # Slice out header
    red_events = red_events[1:]

    def add_ts(d, val, ts):
        # val = (src, dst)
        val = (val[1], val[2])
        if val in d:
            d[val].append(ts)
        else:
            d[val] = [ts]

    anom_dict = {}
    for event in red_events:
        tokens = event.split(',')
        ts = int(tokens.pop(0))
        add_ts(anom_dict, tokens, ts)

    return anom_dict 

def is_anomalous(d, src, dst, ts):
    if ts < 150885 or (src, dst) not in d:
        return False 

    times = d[(src,dst)]
    for time in times:
        # Mark true if node appeared in a comprimise
        # in the last 24 hrs (as was done by Nethawk)
        if ts == time:
            return True

    return False 


def split():
    anom_dict = mark_anoms()

    last_time = 1
    cur_time = 0

    f_in = open(SRC,'r')
    f_out = open(DST + str(cur_time) + '.txt', 'w+')

    line = f_in.readline() # Skip headers
    line = f_in.readline()

    nmap = {} 
    nid = [0]

    def get_or_add(n):
        if n not in nmap:
            nmap[n] = nid[0]
            nid[0] += 1

        return nmap[n]

    prog = tqdm(desc='Seconds parsed', total=5011199)

    fmt_src = lambda x : \
        x.split('@')[0].replace('$', '')

    fmt_label = lambda ts,src,dst : \
        1 if is_anomalous(anom_dict, src, dst, ts) \
        else 0 

    # Really only care about time stamp, and src/dst computers
    # Hopefully this saves a bit of space when replicating the huge
    # auth.txt flow file
    fmt_line = lambda ts,src,dst : (
        '%s,%s,%s,%s\n' % (
            ts, get_or_add(src), get_or_add(dst), 
            fmt_label(int(ts),src,dst)
        ), 
        int(ts)
    )

    while line:
        # Some filtering for better FPR/less Kerb noise
        if 'NTLM' not in line.upper():
            line = f_in.readline()
            continue

        tokens = line.split(',')
        l, ts = fmt_line(tokens[0], tokens[3], tokens[4])

        if ts != last_time:
            prog.update(ts-last_time)
            last_time = ts

        # After ts progresses at least 10,000 seconds, make a new file
        if ts >= cur_time+DELTA:
            cur_time += DELTA
            f_out.close()
            f_out = open(DST + str(cur_time) + '.txt', 'w+')
        
        f_out.write(l)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    nmap_rev = [None] * (max(nmap.values()) + 1)
    for (k,v) in nmap.items():
        nmap_rev[v] = k

    with open(DST + 'nmap.pkl', 'wb+') as f:
        pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    split()
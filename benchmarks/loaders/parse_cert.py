from dateutil import parser
import glob 

from joblib import Parallel, delayed
import torch
from torch_geometric.data import Data 
from tqdm import tqdm

from .load_data import TData

# Parser for CERT insider threat 
# dataset: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247/1
DATA = '/home/ead/iking5/data/CERT_InsiderTheat/'

def parse_redlog(campaign):
    red_events = set()
    files = glob.glob(DATA+'answers/'+campaign+'**/*.csv')

    for file in files:
        f = open(file,'r')

        line = f.readline()
        while(line):
            tokens = line.split(',')
            red_events.add(tokens[1])
            line = f.readline() 
        f.close() 
    
    return red_events

fmt_ts = lambda x : parser.parse(x).timestamp()
def parse_file(fname, red):
    f = open(fname, 'r')
    line = f.readline() # Skip header
    line = f.readline() 

    prog = tqdm(desc=fname.split('/')[-1])
    y,ts,src,dst,etype = [],[],[],[],[]
    while line:
        eid, t, user, pc, event = line.split(',')
        
        y.append(1 if eid in red else 0)
        ts.append(fmt_ts(t))
        src.append(user)
        dst.append(pc)
        etype.append(event.replace('\n',''))

        prog.update()
        line = f.readline()

    f.close()
    return y,ts,src,dst,etype

def parse_all(campaign='r4.2'):
    red = parse_redlog(campaign)

    device = DATA+campaign+'/device.csv'
    logon = DATA+campaign+'/logon.csv'
    
    results = Parallel(n_jobs=2, prefer='processes')(
        delayed(parse_file)(f, red) for f in [device, logon]
    )

    torch.save(results, DATA+campaign+'_raw.pt')
    return torchify(results, campaign)

def torchify(results, campaign, val_size=0.05):
    '''
    Convert from list to sorted tensor
    '''
    ys,ts,src,dst,etype = [],[],[],[],[]

    for r in results:
        y,t,s,d,e = r 
        ys += y; ts += t; src += s; dst += d; etype += e 

    nmap = dict(); nid=0; x=[]
    for s in src:
        if s not in nmap:
            nmap[s] = nid
            nid += 1
            x.append(0)
    for d in dst:
        if d not in nmap:
            nmap[d] = nid 
            nid += 1
            x.append(1)

    emap = dict(); eid = 0
    for e in etype:
        if e not in emap:
            emap[e] = eid 
            eid += 1
        
    etype = torch.tensor([emap[e] for e in etype])
    ei = torch.tensor([
        [nmap[s] for s in src],
        [nmap[d] for d in dst]
    ])
    
    xs = torch.zeros(len(x),2)
    xs[torch.arange(xs.size(0)),torch.tensor(x)] = 1.

    ys = torch.tensor(ys)
    ts = torch.tensor(ts)
    ts = ts - ts.min()

    ts,order = ts.sort()
    ei = ei[:,order]
    ys = ys[order]
    etype = etype[order]

    eis, ets, ys = split(ei, etype, ts, ys)

    test_starts = 0
    for y in ys:
        if y.sum():
            break 
        else:
            test_starts += 1

    masks = []
    for tr in range(test_starts):
        m = torch.rand(eis[tr].size(1))
        m = m > val_size

        # "Test" is a whole dif dataset 
        # for compatibility, just vector of 1's
        masks.append(
            torch.cat([
                m, ~m, torch.ones(m.size())
            ], dim=0)
        )

    tr = TData(
        eis=eis[:test_starts], 
        ts=ts[:test_starts],
        etype=ets[:test_starts],
        masks=masks,
        x=xs,
        nmap=nmap,
        emap=emap,
    ) 
    torch.save(tr, DATA+campaign+'_tr.pt')
    
    te = TData(
        eis=eis[test_starts:], 
        y=ys[test_starts:], 
        etype=ets[test_starts:],
        x=xs,
        nmap=nmap,
        emap=emap,
    ) 
    torch.save(te, DATA+campaign+'_te.pt')

    return tr, te 

def split(ei, etype, ts, y, snapshot_size=60*60*24):
    '''
    Split data into discrete snapshots (days for now)
    '''
    spans = [0]
    next_segment = snapshot_size

    # Find split points
    for i in range(ts.size(0)):
        if ts[i] >= next_segment:
            spans.append(i)
            next_segment += snapshot_size
    spans.append(ts.size(0))

    # Split into chunks
    eis = []; ets = []; ys = []
    for i in range(len(spans)-1):
        eis.append(ei[spans[i]:spans[i+1]])
        ets.append(etype[spans[i]:spans[i+1]])
        ys.append(y[spans[i]:spans[i+1]])

    return eis, ets, ys 

if __name__ == '__main__':
    campaign = 'r4.2'
    res = torch.load(DATA+campaign+'_raw.pt')
    torchify(res, campaign)
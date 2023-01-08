from dateutil import parser
import glob 

from joblib import Parallel, delayed
import torch
from tqdm import tqdm

# Parser for CERN insider threat 
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

def torchify(results, campaign):
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

    torch.save({
        'ts':ts, 
        'ei':ei, 
        'x': xs,
        'ys':ys, 
        'etype':etype,
        'nmap':nmap,
        'emap':emap
    }, DATA+campaign+'_structured.pt')

    return ts, ei, ys, etype

if __name__ == '__main__':
    parse_all()
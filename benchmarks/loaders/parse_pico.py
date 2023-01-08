import json 
from dateutil import parser
from glob import glob 

DATA = '/home/ead/iking5/data/PicoDomain/Zeek_Logs/'
DAY = {
    0: '2019-07-19/',
    1: '2019-07-20/',
    2: '2019-07-21/'
}

def get_labels():
    allowed = {}
    ip_map = {}

    with open(DATA+'../Allowed_Edges.csv', 'r') as f:
        allowed_txt = f.read()


    # Skip header and trailing \n
    lines = allowed_txt.split('\n')[1:-1]

    for line in lines:
        dst,ip,src = line.split(',')
        
        allowed[src]=dst 
        ip_map[ip]=src 

    # To avoid the note added at the end of the file
    allowed['PFSENSE'] = 'ROOT'
    return allowed, ip_map

ALLOWED, IP_MAP = get_labels()

def parse_file(fname):
    f = open(fname, 'r')

    srcs,dsts = [],[]
    etype = []
    labels = []
    ts = []

    line = f.readline()
    while line:
        d = json.loads(line)
        
        if (dst := d.get('service')):
            service_dst = dst.split('/')
            service = service_dst[0].upper()

            # Lots of noise
            if service not in ['HOST', 'RPCSS', 'RESTRICTEDKRBHOST']:
                line = f.readline()
                continue

            if len(service_dst) > 1:
                dst = service_dst[1]
            else:
                dst = service_dst[0]

            dst = dst.split('.')[0].upper()
            dst = dst.split('$')[0]

            if 'client' in d:
                src = d['client'].split('/')[0].upper().replace('$','')
            else: 
                src = IP_MAP[d['id.orig_h']]

            srcs.append(src)
            dsts.append(dst)
            etype.append(service.upper())
            labels.append(0 if dst == ALLOWED.get(src) else 1)
            ts.append(int(parser.parse(d['ts']).timestamp()))

        line = f.readline()

    f.close()
    return srcs,dsts,etype,labels,ts 


def parse_all(st=0,end=2):
    files = []
    for i in range(st,end+1):
        day = DAY[i]
        fs = glob(DATA+day+'kerb*')

        fs.sort()
        files += fs

    src,dst,etype,labels,ts = [],[],[],[],[]
    for file in files:
        s,d,et,l,t = parse_file(file)
        src+=s; dst+=d; etype+=et; labels+=l; ts+=t
        
    return src,dst,etype,labels,ts

src,dst,etype,ys,_ = parse_all()
[print(src[i],'-(%s)->' % etype[i], dst[i], ys[i], sep='\t') for i in range(len(src))]
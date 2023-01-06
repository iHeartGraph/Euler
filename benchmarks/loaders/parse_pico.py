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
    labels = []
    ts = []

    line = f.readline()
    while line:
        d = json.loads(line)
        
        if (dst := d.get('service')):
            service_dst = dst.split('/')
            service = service_dst[0]

            if service == 'host':
                dst = service_dst[1].split('.')[0].upper()

                if 'client' in d:
                    src = d['client'].split('/')[0].upper()
                else: 
                    src = IP_MAP[d['id.orig_h']]

                srcs.append(src)
                dsts.append(dst)
                labels.append(0 if dst == ALLOWED.get(src) else 1)
                ts.append(int(parser.parse(d['ts']).timestamp()))

        line = f.readline()

    f.close()
    return srcs,dsts,labels,ts 

def parse_all():
    files = []
    for i in range(3):
        day = DAY[i]
        fs = glob(DATA+day+'kerb*')

        fs.sort()
        files += fs

    src,dst,labels,ts = [],[],[],[]
    for file in files:
        s,d,l,t = parse_file(file)
        src+=s; dst+=d; labels+=l; ts+=t

    return src,dst,labels,ts

parse_all()
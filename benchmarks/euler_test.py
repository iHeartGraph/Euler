from copy import deepcopy
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score as auc_score,\
     average_precision_score as ap_score
import torch 
from torch.optim import Adam

import generators as g
import loaders.load_data as ld
from models.euler_serial import EulerGCN
from utils import get_score

torch.set_num_threads(32)

NUM_TESTS = 5
PATIENCE = 100
MAX_DECREASE = 2
TEST_TS = 3

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

def train(model, data, end_tr, epochs=1500, pred=False, nratio=1, lr=0.01):
    print(lr)

    opt = Adam(model.parameters(), lr=lr)
    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()
        zs = None

        # Get embedding        
        zs = model(data.x, data.eis, data.tr)[:end_tr]

        if not pred:
            p,n,z = g.link_detection(data, data.tr, zs, nratio=nratio)
            
        else:
            p,n,z = g.link_prediction(data, data.tr, zs, nratio=nratio)      
        

        loss = model.loss_fn(p,n,z)
        loss.backward()
        opt.step()

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        trloss = loss.item() 
        with torch.no_grad():
            model.eval()
            zs = model(data.x, data.eis, data.tr)[:end_tr]

            if not pred:
                p,n,z = g.link_detection(data, data.va, zs)
                st, sf = model.score_fn(p,n,z)
                sscores = get_score(st, sf)

                print(
                    '[%d] Loss: %0.4f  \n\tSt %s ' %
                    (e, trloss, fmt_score(sscores) ),
                    end=''
                )

                avg = sscores[0] + sscores[1]

            else:    
                dp,dn,dz = g.link_prediction(data, data.va, zs, include_tr=False)
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                dp,dn,dz = g.new_link_prediction(data, data.va, zs)
                dt, df = model.score_fn(dp,dn,dz)
                dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f  \n\tPr  %s  \n\tNew %s' %
                    (e, trloss, fmt_score(dscores), fmt_score(dnscores) ),
                    end=''
                )

                avg = (
                    dscores[0] + dscores[1] 
                )

            if avg > best[0]:
                print('*')
                best = (avg, deepcopy(model))
                no_improvement = 0

            # Log any epoch with no progress on val set; break after 
            # a certain number of epochs
            else:
                print()
                # Though it's not reflected in the code, the authors for VGRNN imply in the
                # supplimental material that after 500 epochs, early stopping may kick in 
                if e > 100:
                    no_improvement += 1
                if no_improvement == PATIENCE:
                    print("Early stopping...\n")
                    break

    return best[1]

@torch.no_grad()
def test(model, data, pred, end_tr):
    model.eval()
    
    # Inductive
    if not pred:
        zs = model(data.x, data.eis, data.tr)[end_tr-1:]
    
    # Transductive
    else:
        zs = model(data.x, data.eis, data.all)[end_tr-1:]

    if not pred:
        zs = zs[1:]
        p,n,z = g.link_detection(data, data.te, zs, start=end_tr)
        t, f = model.score_fn(p,n,z)
        sscores = get_score(t, f)

        print(
            '''
            Final scores: 
                Static LP:  %s
            '''
        % fmt_score(sscores))

        return {'auc': sscores[0], 'ap': sscores[1]}

    else:              
        p,n,z = g.link_prediction(data, data.all, zs, start=end_tr-1)
        t, f = model.score_fn(p,n,z)
        dscores = get_score(t, f)

        p,n,z = g.new_link_prediction(data, data.all, zs, start=end_tr-1)
        t, f = model.score_fn(p,n,z)
        nscores = get_score(t, f)

        print(
            '''
            Final scores: 
                Dynamic LP:     %s 
                Dynamic New LP: %s 
            ''' %
            (fmt_score(dscores),
                fmt_score(nscores))
        )

        return {
            'pred-auc': dscores[0],
            'pred-ap': dscores[1],
            'new-auc': nscores[0], 
            'new-ap': nscores[1],
        }

@torch.no_grad()
def test_cert(model, data, pred, h0):
    model.eval()
    zs = model(data.xs, data.eis, data.all, h_0=h0)
    
    if pred:
        ys = data.y[1:]; zs = zs[:-1]; eis=data.eis[:-1]
    else:
        ys = data.y; eis=data.eis

    ys = torch.cat(ys)
    preds = model.score_all(zs, eis)

    auc = auc_score(ys, preds)
    ap = ap_score(ys, preds)

    print("AUC:\t%f\tAP:\t%f" % (auc,ap)) 
    return {'auc':auc, 'ap':ap}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--predict',
        action='store_true',
        help='Sets model to train on link prediction rather than detection'
    )
    parser.add_argument(
        '--lstm',
        action='store_true'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.02
    )
    parser.add_argument(
        '--hidden', 
        type=int,
        default=32
    )
    parser.add_argument(
        '--embedding',
        type=int,
        default=16
    )
    parser.add_argument(
        '--days',
        type=int, 
        default=1
    )
    return parser.parse_args()

def ndss_benchmarks():
    '''
    0.02 is default as it's the best overall, but for the DBLP dataset, 
    lower LR's (0.005 in the paper) work better for new pred tasks
    Optimal LRs are: 
        +---------+-------+-------+-------+
        | Dataset | Det   | Pred  | New   | 
        +---------+-------+-------+-------+
        | Enron   | 0.02  | 0.02  | 0.2   |
        +---------+-------+-------+-------+
        | FB      | 0.01  | 0.02  | 0.1   |
        +---------+-------+-------+-------+
        | DBLP    | 0.02  | 0.02  | 0.005 | 
        +---------+-------+-------+-------+
    '''
    args = get_args()
    outf = 'euler.txt' 

    for d in ['enron10', 'fb', 'dblp']:
        data = ld.load_vgrnn(d)
        model = EulerGCN(data.x.size(1), 32, 16, lstm=args.lstm)
        end_tr = data.T-TEST_TS

        stats = [
            test(
                train(
                    deepcopy(model), 
                    data, end_tr,
                    pred=args.predict, 
                    lr=args.lr
                ), 
                data, args.predict, end_tr
            ) for _ in range(NUM_TESTS)
        ]

        df = pd.DataFrame(stats)
        print(df.mean()*100)
        print(df.sem()*100)

        f = open(outf, 'a')
        f.write(d + '\n')
        f.write('LR: %0.4f\n' % args.lr)
        f.write(str(df.mean()*100) + '\n')
        f.write(str(df.sem()*100) + '\n\n')
        f.close()


CERT = '/home/ead/iking5/data/CERT_InsiderTheat/'
def cert_benchmark():
    from loaders.parse_cert import quick_build

    args = get_args()
    outf = 'euler_cert.txt'

    quick_build(days=args.days)

    tr = torch.load(CERT+'r4.2_tr.pt')
    te = torch.load(CERT+'r4.2_te.pt')

    model = EulerGCN(
        tr.x.size(1), args.hidden, 
        args.embedding, lstm=args.lstm,
        add_bidirect=True
    )
    stats = []
    for _ in range(NUM_TESTS):
        eval_model = train(
            deepcopy(model), 
            tr, -1,
            pred=args.predict, 
            lr=args.lr
        )
        
        eval_model.eval()
        with torch.no_grad():
            _, h0 = eval_model(tr.x, tr.eis, tr.all, include_h=True)
        stats.append(test_cert(eval_model, te, args.predict, h0))
        

    df = pd.DataFrame(stats)
    print(df.mean()*100)
    print(df.sem()*100)

    f = open(outf, 'a')
    f.write('LR: %0.4f\n' % args.lr)
    f.write('Hidden: %d\nEmb: %d\nSize (days): %d\n' % (args.hidden, args.embedding, args.days))
    f.write(str(df)+'\n')
    f.write(str(df.mean()*100) + '\n')
    f.write(str(df.sem()*100) + '\n\n')
    f.close()

if __name__ == '__main__':
    cert_benchmark()
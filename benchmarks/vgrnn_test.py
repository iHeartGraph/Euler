from copy import deepcopy
import argparse

import pandas as pd
import torch 
from torch.optim import Adam

import generators as g
import loaders.load_vgrnn as vd
from models.vgrnn import VGRNN
from utils import get_score

torch.set_num_threads(8)

NUM_TESTS = 5
PATIENCE = 100
MAX_DECREASE = 2
TEST_TS = 3

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

def train(model, data, epochs=1500, pred=False, nratio=1, lr=0.01):
    print(lr)
    end_tr = data.T-TEST_TS

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

    model = best[1]
    with torch.no_grad():
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


def ndss_benchmarks():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--predict',
        action='store_true',
        help='Sets model to train on link prediction rather than detection'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01
    )

    args = parser.parse_args()
    outf = 'vgrnn.txt' 

    for d in ['enron10', 'fb', 'dblp']:
        data = vd.load_vgrnn(d)
        model = VGRNN(data.x.size(1), 32, 16, pred=args.predict)
        
        stats = [
            train(
                deepcopy(model), 
                data, 
                pred=args.predict, 
                lr=args.lr
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
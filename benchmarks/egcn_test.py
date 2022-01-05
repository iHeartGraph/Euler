from argparse import ArgumentParser
from copy import deepcopy
from types import SimpleNamespace as SN

import pandas as pd
import torch 
from torch.optim import Adam
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.loop import add_remaining_self_loops

import generators as g
import loaders.load_vgrnn as vd
from models.evo_gcn import LP_EGCN_h, LP_EGCN_o
from utils import get_score

torch.set_num_threads(8)

NUM_TESTS = 5
PATIENCE = 100
MAX_DECREASE = 2
TEST_TS = 3

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

def convert_to_dense(data, mask, start=0, end=None):
    end = data.T if not end else end

    adjs = []
    for t in range(start, end):
        ei = data.get_masked_edges(t, mask)
        ei = add_remaining_self_loops(ei, num_nodes=data.num_nodes)[0]
        
        a = to_dense_adj(ei, max_num_nodes=data.num_nodes)[0]
        d = a.sum(dim=1)
        d = 1/torch.sqrt(d) 
        d = torch.diag(d)
        ahat = d @ a @ d

        adjs.append(ahat)

    return adjs, [torch.eye(data.num_nodes) for _ in range(len(adjs))]

def train(model, data, epochs=1500, pred=False, nratio=10, lr=0.01):
    print(lr)
    end_tr = data.T-TEST_TS

    tr_adjs, tr_xs = convert_to_dense(data, data.TR, end=end_tr)
    opt = Adam(model.parameters(), lr=lr)

    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()

        # Get embedding   
        zs = model(tr_adjs, tr_xs)

        if not pred:
            p,n,z = g.link_detection(data, data.tr, zs, include_tr=False, nratio=nratio)
            
        else:
            p,n,z = g.link_prediction(data, data.tr, zs, include_tr=False, nratio=nratio)      
        
        loss = model.loss_fn(p,n,z)
        loss.backward()
        opt.step()

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        trloss = loss.item() 
        with torch.no_grad():
            model.eval()
            zs = model(tr_adjs, tr_xs)

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
                dp,dn,dz = g.link_prediction(data, data.va, zs)
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                dp,dn,dz = g.new_link_prediction(data, data.va, zs)
                dt, df = model.score_fn(dp,dn,dz)
                dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f  \n\tDet %s  \n\tNew %s' %
                    (e, trloss, fmt_score(dscores), fmt_score(dnscores)),
                    end=''
                )

                avg = (
                    dscores[0] + dscores[1] 
                )
            
            if avg > best[0]:
                best = (avg, deepcopy(model))
                no_improvement = 0
                print('*')
            else:
                # Though it's not reflected in the code, the authors for VGRNN imply in the
                # supplimental material that after 500 epochs, early stopping may kick in 
                print()
                if e > 100:
                    no_improvement += 1
                if no_improvement == PATIENCE:
                    print("Early stopping...\n")
                    break


    model = best[1]
    with torch.no_grad():
        model.eval()
        
        adjs, xs = convert_to_dense(data, data.TR if not pred else data.ALL)
        zs = model(adjs, xs)[end_tr:]

        if not pred:
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
            p,n,z = g.link_prediction(data, data.all, zs, start=end_tr)
            t, f = model.score_fn(p,n,z)
            dscores = get_score(t, f)

            p,n,z = g.new_link_prediction(data, data.all, zs, start=end_tr)
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


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument(
        '-p', '--predict',
        action='store_true'
    )
    ap.add_argument(
        '-i', '--innerproduct',
        action='store_true'
    )
    ap.add_argument(
        '--lr',
        type=float,
        default=0.01
    )
    ap.add_argument(
        '-o',
        action='store_true'
    )
    args = ap.parse_args()

    m_constructor = LP_EGCN_o if args.o else LP_EGCN_h
    outf = 'egcn_o.txt' if args.o else 'egcn_h.txt'

    for d in ['enron10', 'fb', 'dblp']:
        data = vd.load_vgrnn(d)
        model = m_constructor(data.num_nodes, 32, 16, inner_prod=args.innerproduct)
        stats = [train(deepcopy(model), data, pred=args.predict, lr=args.lr) for _ in range(NUM_TESTS)]

        df = pd.DataFrame(stats)
        print(df.mean()*100)
        print(df.sem()*100)

        f = open(outf, 'a')
        f.write(d + '\n')
        f.write('===== LR %0.4f; Using inner prod? %s =====\n' % (args.lr, args.innerproduct))
        f.write(str(df.mean()*100) + '\n')
        f.write(str(df.sem()*100) + '\n\n')
        f.close()
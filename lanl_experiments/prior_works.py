import sys
from copy import deepcopy
from time import time
from numpy.core.fromnumeric import argmax

import pandas as pd 
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    f1_score, average_precision_score as ap_score
import torch 
from torch import nn 
from torch.optim import Adam 
from torch_geometric.nn import GCNConv

from models.recurrent import GRU 
from models.serial_models import \
    VGRNN, \
    SparseEGCN_H as EGCN_H, \
    SparseEGCN_O as EGCN_O
import loaders.load_lanl as ld 
from loaders.tdata import TData
from utils import get_optimal_cutoff, get_score

class TGCN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(TGCN, self).__init__()

        # Field for anom detection
        self.cutoff = None

        # Topological encoder
        self.c1 = GCNConv(x_dim, h_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)
        self.c2 = GCNConv(h_dim, h_dim)
        self.tanh = nn.Tanh() 

        # Temporal encoder
        self.rnn = GRU(h_dim, h_dim, z_dim)

    
    def forward(self, data, mask, h0=None):
        zs = []
        for i in range(data.T):
            zs.append(
                self.forward_once(
                    data.ei_masked(mask, i),
                    data.ew_masked(mask, i),
                    data.xs
                )
            )

        return self.rnn(torch.stack(zs), h0, include_h=True)
        
    def forward_once(self, ei, ew, x):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)
        return self.tanh(x)


    def decode(self, e, zs):
        src,dst = e 
        return torch.sigmoid(
            (zs[src] * zs[dst]).sum(dim=1)
        )

    def bce(self, pos, neg):
        EPS = 1e-8
        ps = -torch.log(pos+EPS).mean()
        ns = -torch.log(1-neg+EPS).mean()

        return (ps + ns) * 0.5

    def calc_loss(self, p,n,zs):
        tot_loss = torch.zeros(1)

        for i in range(len(zs)):
            tot_loss += self.bce(
                self.decode(p[i], zs[i]),
                self.decode(n[i], zs[i])
            )

        return tot_loss.true_divide(len(zs))

    def calc_scores(self, p,n,zs):
        ps, ns = [], []
        for i in range(len(zs)):
            ps.append(self.decode(p[i], zs[i]))
            ns.append(self.decode(n[i], zs[i]))
        
        return torch.cat(ps, dim=0), torch.cat(ns, dim=0)


EPOCHS = 1500
PATIENCE = 5
MIN = 1

def train(model: nn.Module, data: TData):
    opt = Adam(model.parameters(), lr=0.001)

    best = (None, 0)
    no_progress = 0
    times = []
    print("Starting training")
    
    for e in range(EPOCHS):
        model.train()
        opt.zero_grad()
        
        start_t = time()
        args = model.forward(data, TData.TRAIN)    
        zs = args[0] if isinstance(model, VGRNN) else args
        
        p = [data.ei_masked(TData.TRAIN, i) for i in range(data.T)]
        n = data.get_negative_edges(TData.TRAIN, nratio=10)
        loss = model.calc_loss(p,n,zs)

        loss.backward()
        opt.step() 
        elapsed = time() - start_t
        times.append(elapsed)

        print("[%d] Loss: %0.4f\t%0.4fs" % (e, loss.item(), elapsed))

        model.eval()
        with torch.no_grad():
            args = model.forward(data, TData.TRAIN)
            zs = args[0] if isinstance(model, VGRNN) else args
            
            p = [data.ei_masked(TData.VAL, i) for i in range(data.T)]
            n = data.get_negative_edges(TData.VAL, nratio=10)
            p,n = model.calc_scores(p,n,zs)

            auc,ap = get_score(p,n)
            print("\tVal  AUC: %0.4f  AP: %0.4f" % (auc,ap), end='')

            tot = auc+ap
            if tot > best[1]:
                no_progress = 0
                best = (deepcopy(model), tot)
                print("*")
            else:
                print()
                if e >= MIN:
                    no_progress += 1 

            if no_progress == PATIENCE:
                print("Early stopping!")
                break 

    model = best[0]
    args = model.forward(data, TData.ALL)
    h0 = args[1] if isinstance(model, VGRNN) else None 
    tpe = sum(times)/len(times)
    print("TPE: %0.4f" % tpe)

    return model, h0, tpe


def find_cutoff(model: nn.Module, data: TData, h0: torch.Tensor):
    p = data.eis
    n = data.get_negative_edges(TData.ALL, nratio=10)

    model.eval()
    with torch.no_grad():
        if isinstance(model, VGRNN):
            zs, h0 = model.forward(data, TData.ALL, h0=h0)
        else: 
            zs = model.forward(data, TData.ALL)
            h0 = torch.zeros((1))
        
        if model.pred:
            p = p[1:]
            n = n[1:]
            zs = zs[:-1]
        
        p,n = model.calc_scores(p,n,zs)

    # Note: this is I think the only instance where fpr weight
    # is not 0.6. 0.5 on VGRNN is optimal
    model.cutoff = get_optimal_cutoff(p,n,fw=0.55)
    return h0 


def test(model: nn.Module, data: TData, h0: torch.Tensor):
    pred = 1 if model.pred else 0

    model.eval()
    with torch.no_grad():
        if isinstance(model, VGRNN):
            zs, _ = model.forward(data, TData.ALL, h0=h0)
        else:
            zs = model.forward(data, TData.ALL)

        scores = torch.cat(
            [
                model.decode(
                    data.eis[i+pred][0],
                    data.eis[i+pred][1], 
                    zs[i]
                )
                for i in range(data.T-pred)
            ],
            dim=0
        )

    # Unweighted scores
    y = torch.cat(data.ys[pred:])
    y_hat = torch.zeros(scores.size(0))
    y_hat[scores <= model.cutoff] = 1

    uw_tpr = y_hat[y==1].mean() * 100
    uw_fpr = y_hat[y==0].mean() * 100
    
    uw_tp = y_hat[y==1].sum()
    uw_fp = y_hat[y==0].sum()
    
    uw_f1 = f1_score(y, y_hat)
    uw_auc = auc_score(y, 1-scores)
    uw_ap = ap_score(y, 1-scores)
   
    # Weighted Scores
    weights = torch.cat(data.cnt[pred:])

    alerts = y_hat * weights
    tp = alerts[y==1].sum()
    fp = alerts[y==0].sum()

    tpr = tp / weights[y==1].sum() 
    fpr = fp / weights[y==0].sum() 

    f1 = f1_score(y, y_hat, sample_weight=weights)
    auc = auc_score(y, 1-scores, sample_weight=weights)
    ap = ap_score(y, 1-scores, sample_weight=weights)

    print("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f" % f1)
    print("AUC: %0.4f" % auc)
    print("AP: %0.8f" % ap)

    return {
        'tpr': tpr, 'fpr': fpr,
        'tp': tp, 'fp': fp,
        'f1': f1, 'auc': auc, 'ap': ap,
        'uw_tpr': uw_tpr, 'uw_fpr': uw_fpr,
        'uw_tp': uw_tp, 'uw_fp': uw_fp, 
        'uw_f1': uw_f1, 'uw_auc': uw_auc, 
        'uw_ap': uw_ap
    }

DELTA=  int(60 * 0.125) if len(sys.argv) < 4 \
        else int(60* float(sys.argv[3]))
TR_START=0
TR_END=ld.DATE_OF_EVIL_LANL-DELTA*3 

VAL_START=TR_END
VAL_END=ld.DATE_OF_EVIL_LANL

TE_START=ld.DATE_OF_EVIL_LANL-DELTA 
TE_END = 5011199 

LOADER = ld.load_lanl_dist 
OUT_F = 'results/vgrnn_updated.txt'

def run_all(is_pred, iter):
    print() 
    print(iter)
    print() 

    data = LOADER(8, start=TR_START, end=TR_END, delta=DELTA)
    model = MODEL(data.xs.size(1), 32, 16, pred=is_pred)

    st = time()
    model, h0, tpe = train(model, data)
    tr_time = time() - st

    data = LOADER(8, start=VAL_START, end=VAL_END, delta=DELTA)
    h0 = find_cutoff(model, data, h0)

    data = LOADER(8, start=TE_START, end=TE_END, delta=DELTA, is_test=True)
    stats = test(model, data, h0)
    stats['tpe'] = tpe 
    stats['tr_time'] = tr_time

    return stats

MODEL = VGRNN if len(sys.argv) == 1 \
        else EGCN_H if 'H' in sys.argv[1].upper() \
        else EGCN_O if 'O' in sys.argv[1].upper() \
        else VGRNN

PRED =  False if (len(sys.argv) <= 2) else \
        True if 'D' in sys.argv[2].upper() else \
        False 

if __name__ == '__main__':
    torch.set_num_threads(8)
    stats = [run_all(PRED, i) for i in range(5)]
    stats = pd.DataFrame(stats)
    mean = stats.mean().to_csv().replace(',', '\t')
    full = stats.to_csv(index=False, header=False).replace(",", ', ')

    with open(OUT_F, 'a+') as f:
        f.write("%s is pred: %s\n" % (str(MODEL), PRED))
        f.write("Delta: %0.3f" % (DELTA/60))
        f.write(str(mean)+'\n')
        f.write(str(full)+'\n\n')
    
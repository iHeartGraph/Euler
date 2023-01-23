from copy import deepcopy

import torch 
from torch import nn 
from torch.autograd import Variable
from torch.distributed import rpc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.nn import MessagePassing

from .embedders import GCN 
from .euler_interface import Euler_Encoder, Euler_Recurrent
from .utils import _remote_method, _remote_method_async, _param_rrefs

import torch 
from torch import nn 
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops as arsl
from torch_sparse import SparseTensor 

import torch_cluster # Required import for the below method to work
random_walk = torch.ops.torch_cluster.random_walk

class SampleConv(nn.Module):
    '''
    Only performs message passing between a random selection of 1-hop
    neighbors. Otherwise equivilant to a GCN conv assuming aggr='mean'
    '''
    def __init__(self, in_feats, out_feats, n_samples=5, aggr='mean', always_sample=False):
        super(SampleConv, self).__init__()

        self.lin = nn.Linear(in_feats, out_feats)
        self.mp = MessagePassing(aggr=aggr)

        self.n = n_samples 
        self.always_sample = always_sample
        self.warnings = set()


    def forward(self, x, ei, edge_weight=None):
        '''
        TODO impliment edge weight; for now, just ignore
        '''
        x = self.lin(x)

        # Only sample during training. Use all edges for evaluation
        if self.training or self.always_sample:
            ei = self.sample(ei)
        
        # Need to invert for propagation if sparse
        if type(ei) == SparseTensor:
            ei = ei.t()

        return self.mp.propagate(ei, x=x, size=None)


    def sample(self, ei, n_samples=None):
        n_samples = self.n if n_samples is None else n_samples 
        batch = torch.tensor(list(range(ei.max())))
        loops = batch.clone()

        # Assumes self loops already in 
        if type(ei) == SparseTensor:
            row, col, _ = ei.csr() 
        
        else:
            ei = arsl(ei)[0]
            row, col, _ = SparseTensor.from_edge_index(ei).csr()
            self.warn('sp', "It is recommended to input edge indices as a torch_sparse.SparseTensor type")

        batch = batch.repeat(n_samples)
        samples = random_walk(row, col, batch, 1, 1., 1.)[0][:, 1]
        
        # Add explicit self-loops to make sure result is 1/|N+1| (n + avg(N(n)))
        batch = torch.cat([batch, loops])
        samples = torch.cat([samples, loops])

        return torch.stack([batch, samples])


    def warn(self, key, msg):
        if key not in self.warnings:
            print(msg)
            self.warnings.add(key)


class SampleMean(SampleConv):
    def __init__(self, n_samples=5, aggr='mean', always_sample=False):
        super().__init__(1,1, n_samples=n_samples, aggr=aggr, always_sample=always_sample)
        
        # Just remove the linear layer and have it take the mean
        # of a random sample of node neighbors
        self.lin = nn.Identity()


# From Ramesh's paper, this is f(u,v,N(u))
class TEdgeAnoms(nn.Module):
    def __init__(self, embed_dim, num_nodes, n_samples):
        super().__init__()

        self.H = SampleMean(n_samples=n_samples)
        self.W = nn.Linear(embed_dim, num_nodes)
        self.sm = nn.Softmax(dim=1)
        
        # Computes a softmax on inputs, so sm is only used 
        # for scoring
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, ei, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self.inner_forward(x, ei)
        return self.inner_forward(x, ei)

    def inner_forward(self, x, ei):
        H = self.H(x,ei)
        distro = self.W(H)

        # Don't really need to return the full W matrix
        # Instead calculate loss on samples of true positives
        if self.training:
            src, dst = ei 
            return H, self.ce_loss(
                distro[src],
                dst 
            )
        else:
            return H, None


    def score(self, H, ei):
        '''
        Scores all edges in ei given a precalculated
        H matrix (possibly made w missing edges)

        Only called in eval so never need grads
        '''
        with torch.no_grad():
            distros = self.sm(self.W(H))
            src,dst = ei

            src_score = distros[src, dst]
            dst_score = distros[dst, src]

            return (src_score+dst_score)*0.5



'''
Attempting strat from this paper:
Unified Graph Embedding-based Anomalous Edge Detection
https://ieeexplore.ieee.org/document/9206720
'''
class TEdgeConv(GCN):
    def __init__(self, data_load, data_kws, h_dim, z_dim, n_samples=5):
        super().__init__(data_load, data_kws, h_dim, z_dim)
        
        # This is a hacky solution. TODO figure out how to tell edge conv number of 
        # output dimensions from GRU
        self.anom_detector = TEdgeAnoms(z_dim//2, self.data.num_nodes, n_samples)

    def score(self, H, ei):
        return self.anom_detector.score(H, ei)


def tedge_rref(loader, kwargs, h_dim, z_dim, **kws):
    return TEdgeEncoder(
        TEdgeConv(loader, kwargs, h_dim, z_dim)
    )

class TEdgeEncoder(Euler_Encoder):
    def detect_anoms(self, zs, partition, no_grad):
        H = []
        tot_loss = torch.zeros((1))
        
        for i in range(self.module.data.T):
            ei = self.module.data.ei_masked(partition, i)
            h, loss = self.module.anom_detector(zs[i], ei, no_grad=no_grad)
            H.append(h)

            if not loss is None:
                tot_loss += loss

        return torch.stack(H), tot_loss.true_divide(len(H))

    def score_edges(self, H, partition, nratio):
        n = self.module.data.get_negative_edges(partition, nratio)

        p_scores = []
        n_scores = []

        for i in range(self.module.data.T):
            p = self.module.data.ei_masked(partition, i)
            if p.size(1) == 0:
                continue

            p_scores.append(self.module.score(H[i], p))
            n_scores.append(self.module.score(H[i], n[i]))

        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)

        return p_scores, n_scores

    def decode_all(self, H, unsqueeze=False):
        '''
        Given node embeddings, return edge likelihoods for 
        all subgraphs held by this model
        For static model, it's very simple. Just return the embeddings
        for ei[n] given zs[n]

        zs : torch.Tensor
            A T x d x N tensor of node embeddings generated by the models, 
            it is safe to assume z[n] are the embeddings for nodes in the 
            snapshot held by this model's TGraph at timestep n
        '''
        assert not H.size(0) < self.module.data.T, \
            "%s was given fewer embeddings than it has time slices"\
            % rpc.get_worker_info().name

        assert not H.size(0) > self.module.data.T, \
            "%s was given more embeddings than it has time slices"\
            % rpc.get_worker_info().name

        preds, ys, cnts = [], [], []
        for i in range(self.module.data.T):
            preds.append(
                self.module.score(H[i], self.module.data.eis[i])
            )

            ys.append(self.module.data.ys[i])
            cnts.append(self.module.data.cnt[i])

        return preds, ys, cnts

    def calc_loss(self, z, partition, nratio):
        '''
        Sum up all of the loss per time step, then average it. For some reason
        this works better than running score edges on everything at once. It's better
        to run BCE per time step rather than all at once

        z : torch.Tensor
            A T x d x N tensor of node embeddings generated by the models, 
            it is safe to assume z[n] are the embeddings for nodes in the 
            snapshot held by this model's TGraph at timestep n
        partition : int 
            An enum representing if this is training/validation/testing for 
            generating negative edges 
        nratio : float
            The model samples nratio * |E| negative edges for calculating loss
        '''
        tot_loss = torch.zeros(1)
        ns = self.module.data.get_negative_edges(partition, nratio)

        for i in range(len(z)):
            ps = self.module.data.ei_masked(partition, i)
            
            # Edge case. Prevents nan errors when not enough edges
            # only happens with very small timewindows 
            if ps.size(1) == 0:
                continue

            tot_loss += self.bce(
                self.decode(ps, z[i]),
                self.decode(ns[i], z[i])
            )

        return tot_loss.true_divide(len(z))


class TEdgeRecurrent(Euler_Recurrent):
    def forward(self, mask_enum, include_h=False, h0=None, no_grad=False):
        if include_h:
            z, h = super().forward(mask_enum, include_h=include_h, h0=h0, no_grad=no_grad)
        else:
            z = super().forward(mask_enum, include_h=include_h, h0=h0, no_grad=no_grad)

        # Train the anomaly detector on the output of the embedder at the same time 
        # note the Variable though; loss here won't backprop into the GNN/RNN
        futs = []
        start = 0
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    TEdgeEncoder.detect_anoms,
                    self.gcns[i],
                    Variable(z[start : end]), 
                    mask_enum,
                    no_grad
                )
            )
            start = end 

        H, loss = [], torch.zeros(1)
        for f in futs:
            obj = f.wait()
            H.append(obj[0])
            loss += obj[1]
        
        self.H = H 
        self.anom_loss = loss 

        if include_h:
            return z, h 
        else: 
            return z

    def score_all(self, *args, unsqueeze=False):
        '''
        Has the distributed models score and label all of their edges
        Sends workers embeddings such that H[n] is used to reconstruct graph at 
        snapshot n

        H : torch.Tensor 
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        '''
        futs = [
            _remote_method_async(
                TEdgeEncoder.decode_all,
                self.gcns[i],
                self.H[i],
                unsqueeze=unsqueeze
            )
        for i in range(self.num_workers) ]

        obj = [f.wait() for f in futs]
        scores, ys, cnts = zip(*obj)
        
        # Compress into single list of snapshots
        scores = sum(scores, [])
        ys = sum(ys, [])
        cnts = sum(cnts, [])

        return scores, ys, cnts


    def loss_fn(self, zs, partition, nratio=1):
        '''
        Runs NLL on each worker machine given the generated embeds
        Sends workers embeddings such that zs[n] is used to reconstruct graph at 
        snapshot n

        zs : torch.Tensor 
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        partition : int
            enum representing train, validation, test sent to workers
        nratio : float
            The workers sample nratio * |E| negative edges for calculating loss
        '''
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    TEdgeEncoder.calc_loss,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )
            start = end 

        tot_loss = torch.zeros(1)
        for f in futs:
            tot_loss += f.wait()

        return [tot_loss.true_divide(self.num_workers), 
                self.anom_loss.true_divide(self.num_workers)]
        

    def score_edges(self, zs, partition, nratio=1):
        '''
        Gets edge scores from dist modules, and negative edges. 
        Sends workers embeddings such that zs[n] is used to reconstruct graph at 
        snapshot n

        zs : torch.Tensor 
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        partition : int
            enum representing train, validation, test sent to workers
        nratio : float
            The workers sample nratio * |E| negative edges for calculating loss
        '''
    
        futs = [
            _remote_method_async(
                TEdgeEncoder.score_edges,
                self.gcns[i],
                self.H[i], 
                partition, nratio
            )
        for i in range(self.num_workers) ]

        pos, neg = zip(*[f.wait() for f in futs])
        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)
    
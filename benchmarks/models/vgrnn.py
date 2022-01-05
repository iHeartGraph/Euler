import torch
from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv

from .euler import GAE
from .grnn import GraphGRU
from .loss_fns import full_adj_nll

# This file contains the VGRNN class, which is updated to use
# Pytorch Geometric since the original uses older, depreciated 
# (slower) functions. Used for speed tests rather than evaluation
# as we just use what the authors reported at face value

class VGAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, embed_dim):
        super(VGAE, self).__init__()

        self.c1 = GCNConv(x_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)

        self.mean = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.std = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        
        self.soft = nn.Softplus()

    def forward(self, x, ei, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        # x = self.drop(x)
        
        mean = self.mean(x, ei)
        if self.eval:
            return mean, torch.zeros((1))

        std = self.soft(self.std(x, ei))

        z = self._reparam(mean, std)
        kld = 0.5 * torch.sum(torch.exp(std) + mean**2 - 1. - std)

        return z, kld

    def _reparam(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1)
        return eps1.mul(std).add_(mean)

class VGAE_Prior(VGAE):
    def forward(self, x, ei, pm, ps, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        
        mean = self.mean(x, ei)
        if self.eval:
            return mean, torch.zeros((1))

        std = self.soft(self.std(x, ei))

        z = self._reparam(mean, std)
        kld = self._kld_gauss(mean, std, pm, ps)

        return z, kld

    '''
    Copied straight from the VGRNN code
    '''
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        '''
        # Only take KLD for nodes that exist in this timeslice 
        # (Assumes nodes with higher IDs appear later in the timeline
        # and that once a node appears it never dissapears. A lofty assumption,
        # I know, but this is what the authors did and it seems to work)

        (makes no difference, just slows down training so removed)

        mean_1 = mean_1[:num_nodes]
        mean_2 = mean_2[:num_nodes]
        std_1 = std_1[:num_nodes]
        std_2 = std_2[:num_nodes]
        '''

        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

'''
Model based on that used by the VGRNN paper
Basically the same, but without the variational part
(though that could easilly be added to make it identical)
'''
class GAE_RNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, grnn=True, variational=True, adj_loss=False):
        
        super(GAE_RNN, self).__init__()

        self.h_dim = h_dim 
        self.grnn = grnn

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU()
        )

        self.encoder = GAE(
            h_dim*2, 
            embed_dim=z_dim, 
            hidden_dim=h_dim
        ) if not variational else VGAE(
            h_dim*2, 
            embed_dim=z_dim,
            hidden_dim=h_dim 
        )

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU()
        )

        self.recurrent = nn.GRUCell(
            h_dim*2, h_dim
        ) if not grnn else GraphGRU(
            h_dim*2, h_dim
        )

        self.variational = variational
        self.kld = torch.zeros((1))
        self.adj_loss = adj_loss

    '''
    Iterates through list of xs, and eis passed in (if dynamic_feats is false
    assumes xs is a single 2d tensor that doesn't change through time)
    '''
    def forward(self, xs, eis, mask_fn, ews=None, start_idx=0):
        zs = []
        h = None 
        self.kld = torch.zeros((1))

        for i in range(len(eis)):
            ei = mask_fn(start_idx + i)
            h,z = self.forward_once(xs, ei, h)
            zs.append(z)

        return torch.stack(zs)


    '''
    Runs net for one snapshot
    '''
    def forward_once(self, x, ei, h):
        if type(h) == type(None):
            h = torch.zeros((x.size(0), self.h_dim))

        x = self.phi_x(x)
        gcn_x = torch.cat([x,h], dim=1)

        if self.variational:
            z, kld = self.encoder(gcn_x, ei)
            self.kld += kld 
        else:
            z = self.encoder(gcn_x, ei)

        h_in = torch.cat([x, self.phi_z(z)], dim=1)

        if self.grnn:
            h = self.recurrent.forward_once(h_in, ei, h)
        else: 
            h = self.recurrent(h_in, h)

        return h, z

    
    '''
    Inner product given edge list and embeddings at time t
    '''
    def decode(self, src, dst, z):
        dot = (z[src] * z[dst]).sum(dim=1)
        return torch.sigmoid(dot) 


    '''
    Given confidence scores of true samples and false samples, return
    neg log likelihood 
    '''
    def calc_loss(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        # KLD loss is always 0 if not variational
        return pos_loss + neg_loss


    '''
    Expects a list of true edges and false edges from each time
    step. Note: edge lists need not be the same length. Requires
    less preprocessing but doesn't utilize GPU/tensor ops as effectively
    as the batched fn  
    '''
    def loss_fn(self, ts, fs, zs):
        tot_loss = torch.zeros((1))
        T = len(ts)

        for i in range(T):
            if not self.adj_loss:
                t_src, t_dst = ts[i]
                f_src, f_dst = fs[i]
                z = zs[i]

                tot_loss += self.calc_loss(
                    self.decode(t_src, t_dst, z),
                    self.decode(f_src, f_dst, z)
                )   
            else:
                tot_loss += full_adj_nll(ts[i], zs[i])

        return tot_loss + self.kld


    '''
    Get scores for true/false embeddings to find ROC/AP scores.
    Essentially the same as loss_fn but with no NLL 
    '''
    def score_fn(self, ts, fs, zs):
        tscores = []
        fscores = []

        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]

            tscores.append(self.decode(t_src, t_dst, z))
            fscores.append(self.decode(f_src, f_dst, z))

        tscores = torch.cat(tscores, dim=0)
        fscores = torch.cat(fscores, dim=0)

        return tscores, fscores


class VGRNN(GAE_RNN):
    def __init__(self, x_dim, h_dim, z_dim, adj_loss=True, pred=True):
        super(VGRNN, self).__init__(x_dim, h_dim, z_dim, grnn=True, variational=True, adj_loss=adj_loss)

        self.encoder = VGAE_Prior(
            h_dim*2, 
            hidden_dim=h_dim,
            embed_dim=z_dim
        )

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim), 
            nn.ReLU()
        )

        self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim), 
            nn.Softplus()
        )

        # Whether we return priors or means during eval
        self.pred = pred

    '''
    Runs net for one timeslice
    '''
    def forward_once(self, x, ei, h):
        if type(h) == type(None):
            h = torch.zeros((x.size(0), self.h_dim))

        x = self.phi_x(x)
        gcn_x = torch.cat([x,h], dim=1)

        prior = self.prior(h)
        prior_std = self.prior_std(prior)
        prior_mean = self.prior_mean(prior)
        
        z, kld = self.encoder(gcn_x, ei, pm=prior_mean, ps=prior_std)
        self.kld += kld 

        h_in = torch.cat([x, self.phi_z(z)], dim=1)
        h = self.recurrent.forward_once(h_in, ei, h)

        # Regardless of if self.pred == True Z is means if self.eval == True
        z = prior_mean if self.pred and self.eval else z
        return h, z





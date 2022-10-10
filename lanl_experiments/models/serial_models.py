import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops as add_sl

# Pulled down from https://github.com/IBM/EvolveGCN
from models.egcn_h import EGCN as EGCN_H, GRCU as GRCU_H
from models.egcn_o import EGCN as EGCN_O, GRCU as GRCU_O

'''
Tests on prior works
'''
class VGAE_Prior(nn.Module):
    def __init__(self, x_dim, hidden_dim, embed_dim):
        super(VGAE_Prior, self).__init__()

        self.c1 = GCNConv(x_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)

        self.mean = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.std = GCNConv(hidden_dim, embed_dim, add_self_loops=True)

        self.soft = nn.Softplus()

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


    def _reparam(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1)
        return eps1.mul(std).add_(mean)


'''
Using same GRU as VGRNN paper, where linear layers are replaced with
graph conv layers
'''
class GraphGRU(nn.Module):
    def __init__(self, in_size, hidden_size, n_layers=1, asl=True):
        super(GraphGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # GRU parameters:
        # Update gate
        self.weight_xz = []
        self.weight_hz = []

        # Reset gate
        self.weight_xr = []
        self.weight_hr = []

        # Activation vector
        self.weight_xh = []
        self.weight_hh = []

        for i in range(self.n_layers):
            if i==0:
                self.weight_xz.append(GCNConv(in_size, hidden_size, add_self_loops=asl))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_xr.append(GCNConv(in_size, hidden_size, add_self_loops=asl))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_xh.append(GCNConv(in_size, hidden_size, add_self_loops=asl))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.25)
        

    '''
    Calculates h_out for 1 timestep 
    '''
    def forward_once(self, x, ei, h):
        h_out = None

        for i in range(self.n_layers):
            if i == 0:
                z_g = self.sig(self.weight_xz[i](x, ei) + self.weight_hz[i](h, ei))
                r_g = self.sig(self.weight_xr[i](x, ei) + self.weight_hr[i](h, ei))
                h_hat = self.tanh(self.weight_xh[i](x, ei) + self.weight_hh[i](r_g * h, ei))
                h_out = z_g * h[i] + (1-z_g) * h_hat

            else:
                z_g = self.sig(self.weight_xz[i](h_out, ei) + self.weight_hz[i](h, ei))
                r_g = self.sig(self.weight_xr[i](h_out, ei) + self.weight_hr[i](h, ei))
                h_hat = self.tanh(self.weight_xh[i](h_out, ei) + self.weight_hh[i](r_g * h, ei))
                h_out = z_g * h[i] + (1-z_g) * h_hat

            h_out = self.drop(h_out)

        # Some people save every layer of the GRU but that seems pointless to me.. 
        # but I dunno. I'm breaking with tradition, I guess
        return h_out


    '''
    Calculates h_out for all timesteps. Returns a 
    (t, batch, hidden) tensor

    xs is a 3D batch of features over time
    eis is a list of edge-lists 
    h is the initial hidden state. Defaults to zero
    '''
    def forward(self, xs, eis, mask_fn=lambda x:x, h=None):
        h_out = []
        
        if type(h) == type(None):    
            h = torch.zeros(xs.size(1), self.hidden_size)

        for t in range(len(eis)):
            x = xs[t]
            ei = mask_fn(t)

            h = self.forward_once(x, ei, h)
            h_out.append(h)

        return torch.stack(h_out)

    
'''
Model based on that used by the VGRNN paper
Basically the same, but without the variational part
(though that could easilly be added to make it identical)
'''
class VGRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, grnn=True, pred=True):
        
        super(VGRNN, self).__init__()

        self.h_dim = h_dim 
        self.grnn = grnn

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU()
        )

        self.encoder = VGAE_Prior(
            h_dim*2, 
            embed_dim=z_dim,
            hidden_dim=h_dim 
        )

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU()
        )

        self.recurrent = GraphGRU(
            h_dim*2, h_dim
        )

        self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
        self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        # Whether we return priors or means during eval
        self.pred = pred

        # Holds encoder loss
        self.kld = torch.zeros((1))

    '''
    Iterates through list of xs, and eis passed in (if dynamic_feats is false
    assumes xs is a single 2d tensor that doesn't change through time)
    '''
    def forward(self, data, mask_enum, h0=None):
        zs = []
        h = None if h0 is None else h0
        self.kld = torch.zeros((1))

        for i in range(data.T):
            xs = data.xs if not data.dynamic_feats else data.xs[i]
            ei = data.ei_masked(mask_enum, i)
            
            # Doesn't use weighted edges
            # ew = data.ew_masked(mask_enum, i)

            h,z = self.forward_once(xs, ei, h)
            zs.append(z)

        return torch.stack(zs), h


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

        # Regardless of if self.pred == True Z is means iff self.eval == True
        z = prior_mean if (self.pred and not self.training) else z
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
    def calc_loss_sparse(self, ts, fs, zs):
        t_scores, f_scores = self.score_fn(ts, fs, zs)

        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return pos_loss + neg_loss + self.kld


    '''
    Expects a list of true edges and false edges from each time
    step. Note: edge lists need not be the same length. Requires
    less preprocessing but doesn't utilize GPU/tensor ops as effectively
    as the batched fn  
    '''
    def calc_loss(self, ts, fs, zs):
        return self.calc_loss_sparse(ts, fs, zs)

        # This takes an absurd amount of time
        tot_loss = torch.zeros((1))
        T = len(ts)

        for i in range(T):
            tot_loss += full_adj_nll(ts[i], zs[i])

        return tot_loss + self.kld


    '''
    Same function, I just called it the wrong thing and this is faster
    than actually fixing method signatures
    '''
    def calc_scores(self, ts, fs, zs):
        return self.score_fn(ts, fs, zs)

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


def full_adj_nll(ei, z):
    A = to_dense_adj(ei, max_num_nodes=z.size(0))[0]
    A_tilde = z@z.T

    temp_size = A.size(0)
    temp_sum = A.sum()
    posw = float(temp_size * temp_size - temp_sum) / temp_sum
    norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
    nll_loss_mat = F.binary_cross_entropy_with_logits(input=A_tilde
                                                        , target=A
                                                        , pos_weight=posw
                                                        , reduction='none')
    nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0,1])
    return - nll_loss


class Sparse_GRCU_H(GRCU_H):
    def __init__(self, args):
        super().__init__(args)
        self.mp = MessagePassing(aggr='mean')
    '''
    Updating for a list of edge lists rather than full adj matrix
    We can use the newer GCNConv instead of the manual MM they use
    '''
    def forward(self,data,mask,xs):
        # Masks to boost specific nodes for egcn_h variant
        # I cannot figure out how they used these in their code, and
        # there is no mention of them in the paper. Maybe it's used for
        # node classification? Not sure. 
        mask_list = [torch.zeros((data.num_nodes,1)) for _ in range(data.T)]

        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t in range(data.T):
            node_embs = xs[t]
            Ahat = add_sl(data.ei_masked(mask, t), num_nodes=data.num_nodes)[0] 

            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights,node_embs,mask_list[t])
            
            # Much faster than multiplying by adj 
            node_embs = node_embs.matmul(GCN_weights)
            node_embs = self.activation(
                self.mp.propagate(Ahat, size=None, x=node_embs)
            )

            out_seq.append(node_embs)

        return out_seq


class Sparse_GRCU_O(GRCU_O):
    def __init__(self, args):
        super().__init__(args)
        self.mp = MessagePassing(aggr='mean')
    '''
    Updating for a list of edge lists rather than full adj matrix
    We can use the newer GCNConv instead of the manual MM they use
    Also updated to play nicely with TData objects
    '''
    def forward(self,data,mask,xs):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t in range(data.T):
            node_embs = xs[t]
            Ahat = add_sl(data.ei_masked(mask, t), num_nodes=data.num_nodes)[0] 

            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights)
            
            # Much faster than multiplying by adj 
            node_embs = node_embs.matmul(GCN_weights)
            node_embs = self.activation(
                self.mp.propagate(Ahat, size=None, x=node_embs)
            )

            out_seq.append(node_embs)

        return torch.stack(out_seq)


from types import SimpleNamespace as SN
class SparseEGCN_O(EGCN_O):
    def __init__(self, x_dim, h_dim, z_dim, pred=False):
        # Why do they insist on doing it this way. Fixing it
        args = SN(
            feats_per_node=x_dim,
            layer_1_feats=h_dim,
            layer_2_feats=z_dim
        )
        # RReLU is default in their experiments, keeping it here
        act = torch.nn.RReLU()

        super().__init__(args, act)
        # Doesn't do anything, but makes training this and VGRNN consistant
        self.pred = pred 

        # So my method signature works with copy/pasted code
        feats = [
            x_dim,
            h_dim,
            z_dim
        ]

        # Rewriting with updated GRCU layer
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = SN(
                in_feats=feats[i-1],
                out_feats=feats[i],
                activation=act
            )

            grcu_i = Sparse_GRCU_O(GRCU_args)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    '''
    Updating to work with TData objects (and return all ts embeds)
    '''
    def forward(self, data, mask):
        xs = [data.xs] * data.T
        for unit in self.GRCU_layers:
            xs = unit(data, mask, xs)
        return xs

    # Adding in functions for LP 
    '''
    Inner product given edge list and embeddings at time t
    '''
    def decode(self, src, dst, z):
        dot = (z[src] * z[dst]).sum(dim=1)
        return torch.sigmoid(dot) 


    '''
    Expects a list of true edges and false edges from each time
    step. Note: edge lists need not be the same length. Requires
    less preprocessing but doesn't utilize GPU/tensor ops as effectively
    as the batched fn  
    '''
    def calc_loss(self, ts, fs, zs):
        t_scores, f_scores = self.score_fn(ts, fs, zs)

        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return pos_loss + neg_loss


    '''
    Same function, I just called it the wrong thing and this is faster
    than actually fixing method signatures
    '''
    def calc_scores(self, ts, fs, zs):
        return self.score_fn(ts, fs, zs)

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


class SparseEGCN_H(EGCN_H):
    def __init__(self, x_dim, h_dim, z_dim, pred=False):
        # Why do they insist on doing it this way. Fixing it
        args = SN(
            feats_per_node=x_dim,
            layer_1_feats=h_dim,
            layer_2_feats=z_dim
        )
        # RReLU is default in their experiments, keeping it here
        act = torch.nn.RReLU()

        super().__init__(args, act)
        # Doesn't do anything, but makes training this and VGRNN consistant
        self.pred = pred 

        # So my method signature works with copy/pasted code
        feats = [
            x_dim,
            h_dim,
            z_dim
        ]
        
        # Rewriting with updated GRCU layer
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = SN(
                in_feats=feats[i-1],
                out_feats=feats[i],
                activation=act
            )

            grcu_i = Sparse_GRCU_H(GRCU_args)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))


    '''
    Updating to work with TData objects (and return all ts embeds)
    '''
    def forward(self, data, mask):
        xs = [data.xs] * data.T
        for unit in self.GRCU_layers:
            xs = unit(data, mask, xs)
        return xs

    # Adding in functions for LP 
    '''
    Inner product given edge list and embeddings at time t
    '''
    def decode(self, src, dst, z):
        dot = (z[src] * z[dst]).sum(dim=1)
        return torch.sigmoid(dot) 


    '''
    Expects a list of true edges and false edges from each time
    step. Note: edge lists need not be the same length. Requires
    less preprocessing but doesn't utilize GPU/tensor ops as effectively
    as the batched fn  
    '''
    def calc_loss(self, ts, fs, zs):
        t_scores, f_scores = self.score_fn(ts, fs, zs)

        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return pos_loss + neg_loss


    '''
    Same function, I just called it the wrong thing and this is faster
    than actually fixing method signatures
    '''
    def calc_scores(self, ts, fs, zs):
        return self.score_fn(ts, fs, zs)

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
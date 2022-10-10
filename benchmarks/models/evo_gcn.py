from types import SimpleNamespace as SN

import torch 

# Obtained from https://github.com/IBM/EvolveGCN
from .egcn_h import EGCN as EGCN_h
from .egcn_o import EGCN as EGCN_o

class LP_EGCN_o(EGCN_o):
    def __init__(self, x_dim, h_dim, z_dim, inner_prod=False):
        # Why do they insist on doing it this way. Fixing it
        args = SN(
            feats_per_node=x_dim,
            layer_1_feats=h_dim,
            layer_2_feats=z_dim
        )

        # RReLU is default in their experiments, keeping it here
        super().__init__(args, torch.nn.RReLU())

        # This is how the paper does it, but the experiments show
        # it's not as effective for LP as using inner prod decoding
        if not inner_prod:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(z_dim*2, 1),
                torch.nn.Sigmoid()
            )
        else:
            self.classifier = None
        

    def forward(self,A_list, Nodes_list):
        '''
        Overriding their forward method to return all timesteps
        instead of just the last one
        '''

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list)

        return Nodes_list

    # Copied from Euler
    def calc_loss(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return pos_loss + neg_loss

    def decode(self, src, dst, z):
        if not self.classifier is None:
            catted = torch.cat([z[src], z[dst]], dim=1)
            return self.classifier(catted)
        else:
            dot = (z[src] * z[dst]).sum(dim=1)
            return torch.sigmoid(dot)

    def loss_fn(self, ts, fs, zs):
        tot_loss = torch.zeros((1))
        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]
            
            tot_loss += self.calc_loss(
                self.decode(t_src, t_dst, z),
                self.decode(f_src, f_dst, z)
            )   

        return tot_loss.true_divide(T)

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

class LP_EGCN_h(EGCN_h):
    def __init__(self, x_dim, h_dim, z_dim, inner_prod=False):
        args = SN(
            feats_per_node=x_dim,
            layer_1_feats=h_dim,
            layer_2_feats=z_dim
        )

        # RReLU is default in their experiments, keeping it here
        super().__init__(args, torch.nn.RReLU())

        # This is how the paper does it, but the experiments show
        # it's not as effective for LP as using inner prod decoding
        if not inner_prod:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(z_dim*2, 1),
                torch.nn.Sigmoid()
            )
        else:
            self.classifier = None
        
    def forward(self,A_list, Nodes_list):
        '''
        Overriding their forward method to return all timesteps
        instead of just the last one
        '''
        masks = [torch.zeros(A_list[0].size(0),1) for _ in range(len(A_list))]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list,masks)

        return Nodes_list

    # Copied from Euler
    def calc_loss(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return pos_loss + neg_loss

    def decode(self, src, dst, z):
        if not self.classifier is None:
            catted = torch.cat([z[src], z[dst]], dim=1)
            return self.classifier(catted)
        else:
            dot = (z[src] * z[dst]).sum(dim=1)
            return torch.sigmoid(dot)

    def loss_fn(self, ts, fs, zs):
        tot_loss = torch.zeros((1))
        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]
            
            tot_loss += self.calc_loss(
                self.decode(t_src, t_dst, z),
                self.decode(f_src, f_dst, z)
            )   

        return tot_loss.true_divide(T)

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
    
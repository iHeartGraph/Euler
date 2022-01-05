import torch 
from torch import nn 
from torch.autograd import Variable
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GCNConv 

class EulerModel(nn.Module):
    def __init__(self, static, dynamic):
        super().__init__()

        self.static = static 
        self.dynamic = dynamic 

    def forward(self, data, mask):
        zs = self.static(data, mask)
        preds = self.dynamic(zs)

        return zs, preds

    def loss_fn(self, ps, ns, zs, preds):
        s_loss = self.static.loss_fn(ps, ns, zs)
        d_loss = self.dynamic.loss_fn(zs[1:], preds[:-1])

        return s_loss + d_loss

    def score(self, ps, ns, zs, preds):
        return \
            self.static.score(ps, ns, zs), \
            self.dynamic.score(ps, ns, preds)


# Used by multiple classes
def decode(src, dst, z):
    return torch.sigmoid(
        (z[src] * z[dst]).sum(dim=1)
    )


class Encoder(nn.Module):
    def score(self, ps, ns, zs):
        p_scores = []
        n_scores = []
        
        for i in range(zs.size(0)):
            p_src, p_dst = ps[i]
            p_scores.append(
                decode(p_src, p_dst, zs[i])
            )

            n_src, n_dst = ns[i]
            n_scores.append(
                decode(n_src, n_dst, zs[i])
            )

        return torch.cat(p_scores, dim=0), torch.cat(n_scores, dim=0)
            

    def calc_loss(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return pos_loss + neg_loss

    def loss_fn(self, ps, ns, zs):
        tot_loss = torch.zeros((1))

        for i in range(zs.size(0)):
            p_src, p_dst = ps[i]
            n_src, n_dst = ns[i]
            z = zs[i]

            tot_loss += self.calc_loss(
                    decode(p_src, p_dst, z),
                    decode(n_src, n_dst, z)
                )   

        return tot_loss.true_divide(zs.size(0))


class Recurrent(nn.Module):
    # For now just MSE, but maybe if we make it variational 
    # something like KLD loss would be more effective
    def loss_fn(self, zs, preds):
        return torch.pow(
            (zs - preds), 2
        ).mean()

    def score(self, ps, ns, zs):
        # Predicting future edges, so shift by 1.
        ps, ns, zs = ps[1:], ns[1:], zs[:-1]

        p_scores = []
        n_scores = []
        
        for i in range(zs.size(0)):
            p_src, p_dst = ps[i]
            p_scores.append(
                decode(p_src, p_dst, zs[i])
            )

            n_src, n_dst = ns[i]
            n_scores.append(
                decode(n_src, n_dst, zs[i])
            )

        return torch.cat(p_scores, dim=0), torch.cat(n_scores, dim=0)


# Implimentations of the above
class GAE(Encoder):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        #self.lin = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.c1 = GCNConv(x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        #self.drop = nn.Dropout(0.25)
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True) 

    def forward(self, data, mask):
        zs = []
        for i in range(data.T):
            zs.append(self.inner_forward(data, i, mask))

        return torch.stack(zs)

    def inner_forward(self, data, i, mask):
        if data.dynamic_feats:
            x = data.xs[i]
        else:
            x = data.xs 

        #x = self.lin(x)

        ei = data.ei_masked(mask, i)
        ew = data.ew_masked(mask, i)

        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        #x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)

        return x
    

class GRU(Recurrent):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.rnn = nn.GRU(x_dim, h_dim, num_layers=2)
        self.lin = nn.Linear(h_dim, z_dim)

    def forward(self, zs):
        return self.lin(
            self.rnn(
                torch.tanh(zs)
            )[0]
        )

class LSTM(GRU):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__(x_dim, h_dim, z_dim)
        self.rnn = nn.LSTM(x_dim, h_dim, num_layers=2)

class GAE_GRU(EulerModel):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__(
            GAE(x_dim, h_dim, z_dim),
            GRU(z_dim, h_dim, z_dim)
        )

class GAE_LSTM(EulerModel):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__(
            GAE(x_dim, h_dim, z_dim),
            LSTM(z_dim, h_dim, z_dim)
        )


### Variational models ###
def reparam(mu, logstd):
        return mu + torch.randn_like(logstd) * torch.exp(logstd)

class VEulerModel(EulerModel):
    def __init__(self, static, dynamic):
        super().__init__(static, dynamic)
        self.kld = None
        self.eps = 1e-6

    def kld_loss(self, mu, logstd, p_mu, p_logstd):
        klds = []
        for i in range(len(mu)):
            std_1 = torch.exp(logstd[i])
            std_2 = torch.exp(p_logstd[i])
            
            kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mu - p_mu, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
            klds.append(
                (0.5 / mu.size(1)) * torch.mean(torch.sum(kld_element, dim=1), dim=0)
            )
    
        return torch.cat(klds,dim=0).mean() 

    def forward(self, data, mask):
        mu, logstd = self.static(data, mask)
        zs = reparam(mu, logstd)

        # Make variable s.t. preds doesn't backprop
        # into the encoder. Keep their loss totally seperate
        p_mu, p_logstd = self.dynamic(Variable(zs))
        preds = reparam(p_mu, p_logstd)

        if self.training:
            self.kld = self.kld_loss(Variable(mu[1:]), Variable(logstd[1:]), p_mu[:-1], p_logstd[:-1])
            return zs, preds
        else:
            return mu, p_mu

    def loss_fn(self, ps, ns, zs, preds):
        return self.static.loss_fn(ps, ns, zs) + self.kld

class VGAE(GAE):
    def __init__(self, x_dim, h_dim, z_dim, recons=10):
        super().__init__(x_dim, h_dim, z_dim)
        del self.c2 
        self.mean = GCNConv(h_dim, z_dim)
        self.logstd = GCNConv(h_dim, z_dim)
        self.cnt = 0

        self.kld = None
        self.recon_steps = recons

    def __kld_loss(self, mu, logstd):
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)
        )

    def inner_forward(self, data, i, mask):
        if data.dynamic_feats:
            x = data.xs[i]
        else:
            x = data.xs 

        ei = data.ei_masked(mask, i)
        ew = data.ew_masked(mask, i)

        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        #x = self.drop(x)
        
        mean = self.mean(x, ei, edge_weight=ew)
        logstd = self.logstd(x, ei, edge_weight=ew)

        return mean, logstd

    def forward(self, data, mask):
        mean, logstd = [], []
        kld = []

        for i in range(data.T):
            m, s = self.inner_forward(data, i, mask)
            mean.append(m)
            logstd.append(s)
            kld.append(self.__kld_loss(m,s))

        self.kld = torch.stack(kld).mean()
        return torch.stack(mean), torch.stack(logstd)

    def loss_fn(self, ps, ns, zs):
        return super().loss_fn(ps, ns, zs) + self.kld

class VGRU(GRU):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__(x_dim, h_dim, z_dim)
        del self.lin
        self.mean = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

    def forward(self, zs):
        zs = self.rnn(zs)[0]
        return self.mean(zs), self.std(zs)

class VGAE_VGRU(VEulerModel):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__(
            VGAE(x_dim, h_dim, z_dim),
            VGRU(z_dim, h_dim, z_dim)
        )


class EulerCombo(EulerModel):
    def __init__(self, static, dynamic, h_dim, z_dim):
        super().__init__(static, dynamic)
        self.combiner = nn.Sequential(
            nn.Linear(z_dim*2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim)
        )

    def forward(self, data, mask):
        zs, preds = super().forward(data, mask)
        
        # Shift everything forward 1 so zs[t] and priors[t]
        # are decoded into the same timestep. Zero-pad the
        # initial input
        priors = torch.cat(
            [torch.zeros(preds[0:1].size()),
            preds[:-1]], dim=0
        )

        zs = self.combiner(
            torch.cat([priors, torch.tanh(zs)], dim=-1)
        )

        return zs, preds

'''
Performs about as well as VGRNN (makes sense as it sort of 
is VGRNN but with fewer bells and whistles)
'''
class C_GAE_GRU(EulerCombo):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__(
            GAE(x_dim, h_dim, z_dim),
            GRU(z_dim, h_dim, z_dim),
            h_dim,
            z_dim
        )   

    
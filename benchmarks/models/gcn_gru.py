from math import tan
import torch 
from torch import nn 
from torch_geometric.nn import GCNConv

'''
Using same GRU as VGRNN paper. Updated from their repo to use more
efficient GCNConv class instead of matrix multiplication
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
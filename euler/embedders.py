import torch 
from torch import nn 
from torch.nn import functional as F
from torch.distributed import rpc
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv.message_passing import MessagePassing

from .euler_interface import Euler_Embed_Unit
from .euler_detector import DetectorEncoder
from .euler_predictor import PredictorEncoder


class DropEdge(nn.Module):
    '''
    Implimenting DropEdge https://openreview.net/forum?id=Hkx1qkrKPr
    '''
    def __init__(self, p):
        super().__init__()
        self.p = p 

    def forward(self, ei, ew=None):
        if self.training and self.p > 0:
            mask = torch.rand(ei.size(1))
            if ew is None:
                return ei[:, mask > self.p]
            else:
                return ei[:, mask > self.p], ew[mask > self.p]
                
        if ew is None:
            return ei 
        else: 
            return ei, ew


class GCN(Euler_Embed_Unit):
    '''
    2-layer GCN implimenting the Euler Embed Unit interface
    '''

    def __init__(self, data_load, data_kws, h_dim, z_dim):
        '''
        Constructor for the model 

        parameters
        ----------
        data_load : callable[..., loaders.TGraph]
            Function to load data onto this worker. Must return a loaders.TGraph object
        data_kws : dict 
            Dictionary of keyword args for the loader function 
        h_dim : int 
            The dimension of the hidden layer
        z_dim : int 
            The dimension of the output layer

        attributes
        ----------
        data : loaders.TGraph 
            A TGraph object holding data for all snapshots loaded into this model 
        '''
        super(GCN, self).__init__()

        # Load in the data before initing params
        # Note: passing None as the start or end data_kw skips the 
        # actual loading part, and just pulls the x-dim 
        print("%s loading %s-%s" % (
            rpc.get_worker_info().name, 
            str(data_kws['start']), 
            str(data_kws['end']))
        )

        self.data = data_load(data_kws.pop("jobs"), **data_kws)

        # Params 
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.de = DropEdge(0.8)
    
    
    def inner_forward(self, mask_enum):
        '''
        Override parent's abstract inner_forward method

        mask_enum : int
            enum representing train, validation, test used to mask which 
            edges are sent into the model
        '''
        zs = []
        for i in range(self.data.T):
            # Small optimization. Running each loop step as its own thread
            # is a tiny bit faster. 
            zs.append(
                #torch.jit._fork(self.forward_once, mask_enum, i)
                self.forward_once(mask_enum, i)
            )

        #return torch.stack([torch.jit._wait(z) for z in zs])
        return torch.stack(zs)

    
    def forward_once(self, mask_enum, i):
        '''
        Helper function to make inner_forward a little more readable 
        Just passes each time step through a 2-layer GCN with final tanh activation

        mask_enum : int 
            enum representing train, validation, test 
            used to mask edges passed into model 
        i : int
            The index of the snapshot being processed
        '''
        if self.data.dynamic_feats:
            x = self.data.xs[i]
        else:
            x = self.data.xs 

        ei = self.data.ei_masked(mask_enum, i)
        ew = self.data.ew_masked(mask_enum, i)

        ei, ew = self.de(ei, ew=ew)

        # Simple 2-layer GCN. Tweak if desired
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)

        # Experiments have shown this is the best activation for GCN+GRU
        return self.tanh(x)


# Added dummy **kws param so we can use the same constructor for predictor
def detector_gcn_rref(loader, kwargs, h_dim, z_dim, **kws):
    '''
    Returns a rref to a GCN wrapped in a DetectorEncoder DDP 

    loader : callable[..., loaders.TGraph]
        Function to load data onto this worker. Must return a loaders.TGraph object
    kwargs : dict 
        Dictionary of keyword args for the loader function (must include a field for 'jobs')
    h_dim : int 
        The dimension of the hidden layer
    z_dim : int 
        The dimension of the output layer
    kws : dummy value for matching method signatures
    '''
    return DetectorEncoder(
        GCN(loader, kwargs, h_dim, z_dim)
    )

def predictor_gcn_rref(loader, kwargs, h_dim, z_dim, head=False):
    '''
    Returns a rref to a GCN wrapped in a PredictorEncoder DDP 

    loader : callable[..., loaders.TGraph]
        Function to load data onto this worker. Must return a loaders.TGraph object
    kwargs : dict 
        Dictionary of keyword args for the loader function (must include a field for 'jobs')
    h_dim : int 
        The dimension of the hidden layer
    z_dim : int 
        The dimension of the output layer
    head : If initializing worker0 set true
    '''
    return PredictorEncoder(
        GCN(loader, kwargs, h_dim, z_dim), head
    )


class GAT(GCN):
    '''
    2-layer GAT implimenting the Euler Embed Unit interface. Inherits GCN 
    as the only difference is the forward method, and which submodules are used
    '''

    def __init__(self, data_load, data_kws, h_dim, z_dim, heads=3):
        super().__init__(data_load, data_kws, h_dim, z_dim)

        # Concat=False seems to work best
        self.c1 = GATConv(self.data.x_dim, h_dim, heads=heads, concat=False)
        self.c2 = GATConv(h_dim, z_dim, heads=heads, concat=False)

    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i]
        else:
            x = self.data.xs 

        ei = self.data.ei_masked(mask_enum, i)
        ei = self.de(ei)

        # Only difference is GATs can't handle edge weights
        x = self.c1(x, ei)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei)

        # Experiments have shown this is the best activation for GCN+GRU
        return self.tanh(x)


def detector_gat_rref(loader, kwargs, h_dim, z_dim, **kws):
    return DetectorEncoder(
        GAT(loader, kwargs, h_dim, z_dim)
    )

def predictor_gat_rref(loader, kwargs, h_dim, z_dim, head=False):
    return PredictorEncoder(
        GAT(loader, kwargs, h_dim, z_dim), head
    )


class PoolSAGEConv(MessagePassing):
    '''
    The official PyTorch Geometric package does not actually follow the paper
    This is problematic from both a performance standpoint, and an accuracy one. 
    I have taken it upon myself to build a more correct Maxpool GraphSAGE implientation
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        
        self.aggr_n = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )
        self.e_lin = nn.Linear(out_channels, out_channels)
        self.r_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, ei):
        x_e = self.aggr_n(x)
        x_e = self.propagate(ei, x=x_e, size=None)
        x_e = self.e_lin(x_e)

        x_r = self.r_lin(x)
        
        x = x_r + x_e
        x = F.normalize(x, p=2., dim=-1)
        return x


class SAGE(GAT):
    '''
    2-layer GraphSAGE implimenting the Euler Embed Unit interface. Inherits GAT
    as the only difference is which submodules are used
    '''
    def __init__(self, data_load, data_kws, h_dim, z_dim):
        super().__init__(data_load, data_kws, h_dim, z_dim)

        self.c1 = PoolSAGEConv(self.data.x_dim, h_dim)
        self.c2 = PoolSAGEConv(h_dim, z_dim)


def detector_sage_rref(loader, kwargs, h_dim, z_dim, **kws):
    return DetectorEncoder(
        SAGE(loader, kwargs, h_dim, z_dim)
    )

def predictor_sage_rref(loader, kwargs, h_dim, z_dim, head=False):
    return PredictorEncoder(
        SAGE(loader, kwargs, h_dim, z_dim), head
    )
import time 

import torch 

from euler.embedders import GCN
from euler.euler_detector import DetectorEncoder
from euler.euler_detector import DetectorRecurrent

class TimedEmbedUnit(GCN):
    def forward(self, mask_enum, no_grad):
        st = time.time()
        return super().forward(mask_enum, no_grad), time.time()-st

def timed_gcn_rref(loader, kwargs, h_dim, z_dim, **kws):
    return DetectorEncoder(
        TimedEmbedUnit(loader, kwargs, h_dim, z_dim)
    )
    

class TimedRecurrent(DetectorRecurrent):
    def forward(self, mask_enum, include_h=False, h0=None, no_grad=False):
        '''
        First have each worker encode their data, then run the embeddings through the RNN 

        mask_enum : int
            enum representing train, validation, test sent to workers
        include_h : boolean
            if true, returns hidden state of RNN as well as embeddings
        h0 : torch.Tensor
            initial hidden state of RNN. Defaults to zero-vector if None
        no_grad : boolean
            if true, tells all workers to execute without calculating gradients.
            Used for speedy evaluation
        '''
        full_st = time.time()
        futs = self.encode(mask_enum, no_grad)

        # Run through RNN as embeddings come in 
        # Also prevents sequences that are super long from being encoded
        # all at once. (This is another reason to put extra tasks on the
        # workers with higher pids)
        zs = []
        enc_ts = []
        rnn_ts = []
        net_ts = []

        for f in futs:
            z, t = f.wait()

            st = time.time()
            z, h0 = self.rnn(
                z,
                h0, include_h=True
            )
            rnn_done = time.time() 

            enc_ts.append(enc_ts.append(t))
            rnn_ts.append(rnn_done-st)
            net_ts.append(rnn_done-full_st)
            zs.append(z)


        # May as well do this every time, not super expensive
        self.len_from_each = [
            embed.size(0) for embed in zs
        ]
        zs = torch.cat(zs, dim=0)
        self.z_dim = zs.size(-1)

        return zs, {'enc':enc_ts, 'rnn':rnn_ts, 'full':net_ts}
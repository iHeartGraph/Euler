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
    def forward(self, mask_enum, h0=None, no_grad=False, batched=False):
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

        print(len(futs))

        # Run through RNN as embeddings come in 
        # Also prevents sequences that are super long from being encoded
        # all at once. (This is another reason to put extra tasks on the
        # workers with higher pids)
        zs = []
        enc_ts = []
        rnn_ts = []
        net_ts = []

        if not batched:
            for f in futs:
                waiting = time.time()
                z, t = f.wait()
                waited = time.time()
                net_ts.append(waited-waiting)

                st = time.time()
                z, h0 = self.rnn(
                    z,
                    h0, include_h=True
                )
                rnn_done = time.time() 

                enc_ts.append(t)
                rnn_ts.append(rnn_done-st)
                zs.append(z)

            zs = torch.cat(zs, dim=0)

        else:
            complete = []
            for f in futs:
                waiting = time.time()
                f = f.wait()
                waited = time.time()
                
                complete.append(f)
                net_ts.append(waited-waiting)

            for z,t in complete:
                zs.append(z); enc_ts.append(t)

            st = time.time()
            zs, h0 = self.rnn(
                torch.cat(zs, dim=0),
                h0, include_h=True
            )
            rnn_done = time.time()
            rnn_ts.append(rnn_done-st)

            zs = z 

        self.z_dim = zs.size(-1)

        return zs, {
            'enc':enc_ts, 'rnn':rnn_ts, 'waiting': net_ts,
            'full':sum(net_ts)+sum(rnn_ts), 'if_serial':sum(enc_ts)+sum(rnn_ts)
        }
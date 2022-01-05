import torch 
from torch import nn

'''
Implimenting DropEdge https://openreview.net/forum?id=Hkx1qkrKPr
'''
class DropEdge(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p 

    def forward(self, ei):
        if self.training:
            mask = torch.rand(ei.size(1))
            return ei[:, mask > self.p]
        return ei 
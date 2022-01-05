import torch 
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj

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
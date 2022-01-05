import torch 

'''
Splits edges into 85:5:10 train val test partition
(Following route of VGRNN paper)
'''
def edge_tvt_split(ei):
    ne = ei.size(1)
    val = int(ne*0.85)
    te = int(ne*0.90)

    masks = torch.zeros(3, ne).bool()
    rnd = torch.randperm(ne)
    masks[0, rnd[:val]] = True 
    masks[1, rnd[val:te]] = True
    masks[2, rnd[te:]] = True 

    return masks[0], masks[1], masks[2]

'''
For the cyber data, all of the test set is the latter time
stamps. So only need train and val partitions
'''
def edge_tv_split(ei, v_size=0.05):
    ne = ei.size(1)
    val = int(ne*v_size)

    masks = torch.zeros(2, ne).bool()
    rnd = torch.randperm(ne)
    masks[1, rnd[:val]] = True
    masks[0, rnd[val:]] = True 

    return masks[0], masks[1]
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, \
    roc_curve, precision_recall_curve, auc, f1_score
import torch 

'''
Returns AUC and AP scores given true and false scores
Specialized for when close to 0 implies alert, so scores are 1-score
nscore == Not anomalous
pscore == is anomalous 

(don't get mixed up with positive edge/negative edge)
'''
def get_score(nscore, pscore):
    ntp = pscore.size(0)
    ntn = nscore.size(0)

    score = (1-torch.cat([pscore.detach(), nscore.detach()])).numpy()
    print(score.max())
    print(score.min())
    labels = np.zeros(ntp + ntn, dtype=np.long)
    labels[:ntp] = 1

    ap = average_precision_score(labels, score)
    auc = roc_auc_score(labels, score)

    return [auc, ap]

def get_auprc(probs, y):
    p, r, _ = precision_recall_curve(y, probs)
    pr_curve = auc(r,p)
    return pr_curve

def tf_auprc(t, f):
    nt = t.size(0)
    nf = f.size(0)
    
    y_hat = torch.cat([t,f], dim=0)
    y = torch.zeros((nt+nf,1))
    y[:nt] = 1

    return get_auprc(y_hat, y)

def get_f1(y_hat, y):
    return f1_score(y, y_hat)

'''
Returns the threshold that achieves optimal TPR and FPR
(Can be tweaked to return better results for FPR if desired)

Does this by checking where the TPR and FPR cross each other
as the threshold changes (abs of TPR-(1-FPR))

Please do this on TRAIN data, not TEST
'''
def get_optimal_cutoff(pscore, nscore, fw=0.5):
    ntp = pscore.size(0)
    ntn = nscore.size(0)

    tw = 1-fw

    score = torch.cat([pscore.detach(), nscore.detach()]).numpy()
    labels = np.zeros(ntp + ntn, dtype=np.long)
    labels[:ntp] = 1

    fpr, tpr, th = roc_curve(labels, score)
    fn = np.abs(tw*tpr-fw*(1-fpr))
    best = np.argmin(fn, 0)

    print("Optimal cutoff %0.4f achieves TPR: %0.2f FPR: %0.2f on train data" 
        % (th[best], tpr[best], fpr[best]))
    return th[best]

def dist_AUC(fname):
	'''
	Calcuates AUC score for a CSV of scores,label
	for datasets like LANL which are way to big to 
	hold in memory all at once
	'''
	pass	

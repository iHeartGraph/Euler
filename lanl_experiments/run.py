from argparse import ArgumentParser
import os 

import pandas as pd

import loaders.load_lanl as lanl
import loaders.load_optc as optc
from models.recurrent import GRU, LSTM, EmptyModel
from models.embedders import \
    detector_gcn_rref, detector_gat_rref, detector_sage_rref, \
    predictor_gcn_rref, predictor_gat_rref, predictor_sage_rref
from models.softmax_det import tedge_rref 
from models.softmax_pred import pred_tedge_rref

from spinup import run_all

DEFAULT_TR = {
    'anom_lr': 0.05,
    'epochs': 100,
    'min': 1,
    'nratio': 10,
    'val_nratio': 1
}

OUTPATH = '' # Output folder for results.txt (ending in delimeter)

def get_args():
    global DEFAULT_TR

    ap = ArgumentParser()

    ap.add_argument(
        '-d', '--delta',
        type=float, default=0.5
    )

    ap.add_argument(
        '-w', '--workers',
        type=int, default=8
    )

    ap.add_argument(
        '-T', '--threads',
        type=int, default=1
    )

    ap.add_argument(
        '-e', '--encoder',
        choices=['GCN', 'GAT', 'SAGE'],
        type=str.upper,
        default="GCN"
    )

    ap.add_argument(
        '-r', '--rnn',
        choices=['GRU', 'LSTM', 'NONE'],
        type=str.upper,
        default="GRU"
    )

    ap.add_argument(
        '-H', '--hidden',
        type=int,
        default=32
    )

    ap.add_argument(
        '-z', '--zdim',
        type=int,
        default=16
    )

    ap.add_argument(
        '-n', '--ngrus',
        type=int,
        default=1
    )

    ap.add_argument(
        '-t', '--tests',
        type=int, 
        default=1
    )

    ap.add_argument(
        '-l', '--load',
        action='store_true'
    )

    ap.add_argument(
        '--fpweight',
        type=float,
        default=0.6
    )

    ap.add_argument(
        '--nowrite',
        action='store_true'
    )

    ap.add_argument(
        '--impl', '-i',
        type=str.upper,
        choices=['DETECT', 'PREDICT', 'D', 'P', 'PRED', 'TEDGE', 'TEDGE_PRED'],
        default="DETECT"
    )

    # For future new data sets
    ap.add_argument(
        '--dataset',
        default='LANL', 
        type=str.upper
    )

    ap.add_argument(
        '--lr',
        default=0.005,
        type=float
    )
    ap.add_argument(
        '--patience',
        default=5, 
        type=int
    )

    args = ap.parse_args()
    args.te_end = None
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'

    readable = str(args)
    print(readable)

    model_str = '%s -> %s (%s)' % (args.encoder , args.rnn, args.impl)
    print(model_str)
    
    # Parse dataset info 
    if args.dataset.startswith('L'):
        args.loader = lanl.load_lanl_dist
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None # Computed later
        args.te_times = [(args.tr_end, lanl.TIMES['all'])]
        args.delta = int(args.delta * (60**2))
        args.manual = False 

    elif args.dataset.startswith('O'):
        args.loader = optc.load_optc_dist
        args.tr_start = 0 #optc.TIMES['val_start']
        args.tr_end = optc.TIMES['val_end']
        args.val_times = None #(optc.TIMES['val_start'], optc.TIMES['val_end'])
        args.te_times = [optc.DAY1, optc.DAY2, optc.DAY3, optc.ALL]
        args.delta = int(args.delta * 60)
        args.manual = False 

    else:
        raise NotImplementedError('Only the LANL data set is supported in this release')

    # Convert from str to function pointer
    if args.encoder == 'GCN':
        args.encoder = detector_gcn_rref if args.impl[0] == 'D' \
            else predictor_gcn_rref
    elif args.encoder == 'GAT':
        args.encoder = detector_gat_rref if args.impl[0] == 'D' \
            else predictor_gat_rref
    else:
        args.encoder = detector_sage_rref if args.impl[0] == 'D' \
            else predictor_sage_rref

    # Softmax tests
    if args.impl == 'TEDGE':
        args.encoder = tedge_rref
    if args.impl == 'TEDGE_PRED':
        args.encoder = pred_tedge_rref   

    if args.rnn == 'GRU':
        args.rnn = GRU
    elif args.rnn == 'LSTM':
        args.rnn = LSTM 
    else:
        args.rnn = EmptyModel

    return args, readable, model_str

if __name__ == '__main__':
    args, argstr, modelstr = get_args() 
    DEFAULT_TR['lr'] = args.lr
    DEFAULT_TR['patience'] = args.patience

    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        # Need to tell workers to output in embed dim
        worker_args = [args.hidden, args.zdim]
        rnn_args = [None, None, None]

    stats = [
        run_all(
            args.workers, 
            args.rnn, 
            rnn_args,
            args.encoder, 
            worker_args, 
            args.delta,
            args.load,
            args.fpweight,
            args.impl,
            args.loader, 
            args.tr_start,
            args.tr_end, 
            args.val_times,
            args.te_times,
            DEFAULT_TR
        )
        for _ in range(args.tests)
    ]

    # Don't write out if nowrite
    if args.nowrite:
        exit() 

    f = open(OUTPATH+'results.txt', 'a')
    f.write(str(argstr) + '\n')
    f.write('LR: ' + str(args.lr) + '\n')
    f.write(modelstr + '\n')

    dfs = [pd.DataFrame(s) for s in list(zip(*stats))]
    dfs = pd.concat(dfs, axis=0)

    for m in dfs['Model'].unique():
        df = dfs[dfs['Model'] == m]

        compressed = pd.DataFrame(
            [df.mean(), df.sem()],
            index=['mean', 'stderr']
        ).to_csv().replace(',', '\t') # For easier copying into Excel

        full = df.to_csv(index=False, header=False)
        full = full.replace(',', ', ')

        f.write(m + '\n')
        f.write(str(compressed) + '\n')
        f.write(full + '\n')

    f.close()
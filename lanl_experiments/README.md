# LANL Experiments

This folder contains the code that ran the experiments we report on the LANL data set. 

## Prerequisites
Please download the LANL data set from [https://csr.lanl.gov/data/cyber1/](https://csr.lanl.gov/data/cyber1/). Then, in `./loaders/split.py` please specify the file locations, and output location for the cleaned, split data on lines 13-15 and run it. Finally, on line 17 of `./loaders/load_lanl.py` please specify the location of the LANL files. 

To test the prior works, use the file `prior_works.py`. The command line flags `-v` (default), `-o`, and `-h` evaluate VGRNN, EGCN-O, and EGCN-H respectively.

## Usage
To run experiments on the LANL data set, use `run.py`. Tests used in the paper are automated in `runall_LANL.sh` and `run_delta.sh`. Options include:

    -h, --help    
        Show this help message and exit    
    -d DELTA, --delta DELTA                     (default: 0.5 hrs)    
        The size of snapshots in hours    
    -w WORKERS, --workers WORKERS               (default: 8)    
        The number of worker processes/GNNs    
    -T THREADS, --threads THREADS               (default: 1)    
        The number of threads per worker    
    -e {GCN,GAT,SAGE}, --encoder {GCN,GAT,SAGE} (default: GCN)    
        Which model to use on workers    
    -r {GRU,LSTM,NONE}, --rnn {GRU,LSTM,NONE}   (default: GRU)    
        Which model to use on the leader    
    -H HIDDEN, --hidden HIDDEN                  (default: 32)    
        The number of dimensions for the hidden layer    
    -z ZDIM, --zdim ZDIM                        (default: 16)     
        The number of dimensions for the ouput     
    -n NGRUS, --ngrus NGRUS                     (default: 1)    
        The number of RNNs in the worker    
    -t TESTS, --tests TESTS                     (default: 1)    
        The number of indipendant tests to run    
    -l, --load                                  (default: False)    
        If present, does not train, rather loads the last saved model    
    --fpweight FPWEIGHT                         (default: 0.6)    
        The lambda parameter    
    --nowrite    
        If present, does not write output statistics to a file    
    --impl {DETECT,PREDICT,D,P,PRED}, -i {DETECT,PREDICT,D,P,PRED}
        (default: DETECT)    
        Whether to use the detector or predictor    
    --dataset DATASET                           (default: LANL)
        Unused for now.    
    --lr LR                                     (default: 0.005)    
        Learning rate during training    
    --patience PATIENCE                         (default: 5)    
        The number of epochs with no improvement before training ends    

Note: by default, this script uses ports 22032 and 22204 for RPC communication. 

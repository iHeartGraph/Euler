python run.py -t 5 -d 0.5 -e GCN
python run.py -t 5 -d 0.5 -e GCN -r LSTM
python run.py -t 5 -d 0.5 -e GCN -r NONE
python run.py -t 5 -d 0.5 -e GCN -i PRED
python run.py -t 5 -d 0.5 -e GCN -r LSTM -i PRED
python run.py -t 5 -d 0.5 -e GCN -r NONE -i PRED
python run.py -t 5 -d 0.5 -e SAGE --fpweight 0.5
python run.py -t 5 -d 0.5 -e SAGE -r LSTM --fpweight 0.5
python run.py -t 5 -d 0.5 -e SAGE -r NONE --fpweight 0.5
python run.py -t 5 -d 0.5 -e SAGE -i PRED --fpweight 0.5
python run.py -t 5 -d 0.5 -e SAGE -r LSTM -i PRED --fpweight 0.5
python run.py -t 5 -d 0.5 -e SAGE -r NONE -i PRED --fpweight 0.5
python run.py -t 5 -d 0.5 -e GAT
python run.py -t 5 -d 0.5 -e GAT -r LSTM
python run.py -t 5 -d 0.5 -e GAT -r NONE
python run.py -t 5 -d 0.5 -e GAT -i PRED
python run.py -t 5 -d 0.5 -e GAT -r LSTM -i PRED
python run.py -t 5 -d 0.5 -e GAT -r NONE -i PRED
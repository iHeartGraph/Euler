python run.py -w 16 --dataset OpTC 
python run.py -w 16 --dataset OpTC -e SAGE
python run.py -w 16 --dataset OpTC -e GAT
python run.py -w 16 --dataset OpTC -r LSTM 
python run.py -w 16 --dataset OpTC -r LSTM -e SAGE
python run.py -w 16 --dataset OpTC -r LSTM -e GAT 
python run.py -w 16 --dataset OpTC -i TEDGE
python run.py -w 16 --dataset OpTC -e SAGE -i TEDGE
python run.py -w 16 --dataset OpTC -e GAT -i TEDGE
python run.py -w 16 --dataset OpTC -r LSTM -i TEDGE
python run.py -w 16 --dataset OpTC -r LSTM -e SAGE -i TEDGE
python run.py -w 16 --dataset OpTC -r LSTM -e GAT -i TEDGE
#original seqLen8 hidDim128 nhead4 dropout0.3 n_decLayer6
nohup python main_v2.py --gpu 6,7 --batch_size 512 --lr 0.0001 --save_model False #validation
nohup python main_v2.py --gpu 6,7 --batch_size 512 --lr 0.0001 --save_model False --seq_len 4 --hid_dim 128 --nhead 4 --mode val & #seqLen
nohup python main_v2.py --gpu 6,7 --batch_size 512 --lr 0.0001 --save_model False --seq_len 8 --hid_dim 64 --nhead 4 --mode val & #dim
nohup python main_v2.py --gpu 6,7 --batch_size 512 --lr 0.0001 --save_model False --seq_len 8 --hid_dim 16 --nhead 4 --mode val & #dim
nohup python main_v2.py --gpu 6,7 --batch_size 512 --lr 0.0001 --save_model False --seq_len 8 --hid_dim 128 --nhead 8 --mode val & #nhead
nohup python main_v2.py --gpu 6,7 --batch_size 512 --lr 0.0001 --save_model False --seq_len 8 --hid_dim 128 --n_dec_layer 2 --mode val & #n_decLayer
nohup python main_v2.py --gpu 6,7 --batch_size 512 --lr 0.0001 --save_model False --seq_len 8 --hid_dim 128 --dropout 0.5 --mode val & #dropout


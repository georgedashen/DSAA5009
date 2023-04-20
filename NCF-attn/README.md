# This is the revised version of NCF with attention layers

## Running
For quick start, use `run best_run.sh`. This setting does not guarantee good and correct results.

Note that environments and resources could be different with users, you can `cat best_run.sh` to see the arguments I used or use `python main.py -h` to see all arguments.

For single machine single GPU, I suggest run `python main.py --gpu 0 --batch_size 256 --seq_len 8 --hid_dim 16 --lr 0.0001 --epochs 20` to get proper results. Each epoch need 2-3 minutes to run and model cost around 2G GPU memory.

## Files
`config.py` is for data/output directory and model settings

`data_utils.py` is for data loading

`model.py` is the construction of NCF-attn model

`transformer.py` is the definition of decoder layers with customized forward process

`evaluate.py` is for getting HR and NDCG with test data

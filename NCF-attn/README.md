# This is the revised version of NCF with attention layers

## Running
Make sure to rename the previous converted training set and generated negative sampling val/test set as `taobao.train.rating`, `taobao.val.negative` and `taobao.test.negative`, respectively. The converted training set `train.csv` is provided in this directory.

For quick start, use `python main_v2.py --gpu 0 --trainN 10000 --testN 1000`. This setting only for fast showcase but not guarantee good and correct results.

For single machine single GPU, I suggest using the following command to get proper results. 
```bash
python main_v2.py --gpu 0 --batch_size 256 --seq_len 8 --hid_dim 16 --dropout 0.5 --lr 0.0001 --epochs 20
```

Each epoch need 40-60 minutes to run and model cost around 8G GPU memory. If you have more than one GPU, you can keep a batch size of 256 for each GPU.

To reproduce the results in the report, one can run `sh experiments.sh` for both hyperparameter searching on validation set and final evaluation on test set.

## Files
* `config.py` is for data/output directory and model settings

* `main_v2.py` if for main training process

* `data_utils_v2.py` is for data loading

* `model.py` is the construction of NCF-attn model

* `transformer.py` is the definition of decoder layers with customized forward process

* `evaluate_v2.py` is for getting recall@50 and NDCG@50 with test data

* `experiments.sh` is for fast running directly from Linux bash

# DSAA5009 Final Project
## Group 3 - Recommendation system

## Requirements
All experiments are done on two 3090 GPU with 24G memory.
```
* keras==2.11.0
* numpy==1.21.6
* pandas==1.3.5
* scikit-learn==1.0.2
* scipy==1.7.3
* Theano==1.0.5
* torch==1.10.1+cu111
* tqdm==4.64.1
```

## Negative sampling and data conversion
Noted that in our project, we trained and evaluated our models using negative sampling procedures. For training set, the negative sampling is performed during the training process of NCF/NCF-attn models and to reproduce the result one must use the same random seeding as shown in the code. For non-deep-learning baseline models there is no need to negative sample the training set. For validation/test set, one can use the following code to generate the negative sampling we used. Original data `val_user_consumed.pkl` and `test_user_consumed.pkl` should be placed in the same folder.
```bash
python convert_negative.py
```

## Non-deep-learning baseline models
Baseline experiments provide options for testing random and user-based model on different top n items and k neighbors for users via cosine similarity and different models can be chosen. Default settings are `--k_user 5`, `--top_n 1`, `--model user-based`, without `--test` to indicate use the validation set for evaluations and without `--negative` for not using the negative sampling data.

A quick start for user-based model using all data in test set to evaluate model performance is shown as following:
```bash
python baseline.py --test
```
You can also use `run baseline_experiment_test.sh` to run a set of experiments on test set, which will produce results shown in the report.

For results on negative sampling set, try to use:
```bash
python baseline.py --negative /path/to/negative/sampling/data --test
```
You can also use the provided `run baseline_experiment_negative.sh` to run a set of experiments on test set, which will produce results shown in the report.

## Deep learning baseline models (NeuMF: MLP + GMF)
Most of the code are bollowed from a pytorch version of [NeuMF](https://github.com/guoyang9/NCF), and one can check more details on the original NCF(2017) paper.

[He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).](https://arxiv.org/abs/1708.05031)

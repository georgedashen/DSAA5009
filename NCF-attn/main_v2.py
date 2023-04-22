import os
import time
import argparse
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import model
import config
import evaluate_v2
import data_utils_v2

import csv


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
    type=float, 
    default=1e-5, 
    help="learning rate")
parser.add_argument("--dropout", 
    type=float,
    default=0.3,  
    help="dropout rate")
parser.add_argument("--batch_size", 
    type=int, 
    default=256, 
    help="batch size for training")
parser.add_argument("--epochs", 
    type=int,
    default=20,  
    help="training epoches")
parser.add_argument("--top_k", 
    type=int, 
    default=50, 
    help="compute metrics@top_k")
parser.add_argument("--trainN", 
        type=int,
        default=None,
        help="number of training data to load")
parser.add_argument("--testN", 
        type=int,
        default=None,
        help="number of test data to load")
parser.add_argument("--test_itemN",
        type=int,
        default=500,
        help="number of test items per user to load")
parser.add_argument("--seq_len",
        type=int,
        default=8,
        help="hidden factors seq_len in the model")
parser.add_argument("--hid_dim", 
    type=int,
    default=128, 
    help="hidden factors dimension *seq_len in the model")
parser.add_argument("--feedforward_dim",
        type=int,
        default=2048,
        help="feedforward factors dimension between attention")
parser.add_argument("--fc_dim",
        type=int,
        default=32,
        help="factor dimension  in the predictive head")
parser.add_argument("--n_dec_layer", 
    type=int,
    default=6, 
    help="number of layers in MLP model")
parser.add_argument("--nhead",
        type=int,
        default=4,
        help="number of head in the attention")
parser.add_argument("--rm_self_attn",
        type=bool,
        default=False,
        help="whether to remove self-attention for ablation test")
parser.add_argument("--activation",
        type=str,
        default="relu",
        help="activation function in transformer")
parser.add_argument("--num_ng", 
    type=int,
    default=4, 
    help="sample negative items for training")
parser.add_argument("--seed",
        type=int,
        default=123,
        help="sample part of negative items for testing")
parser.add_argument("--out", 
    default=True,
    help="save model or not")
parser.add_argument("--gpu", 
    type=str,
    default="0",  
    help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

assert args.top_k <= args.test_itemN, "Recommened number should less than known number!"
if len(args.gpu.split(","))>0:
    assert torch.cuda.device_count()>=len(args.gpu.split(",")), "There are not enough GPUs!"


def set_rand_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_rand_seed(args.seed)

############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat, labels  = data_utils_v2.load_all(args.trainN, args.testN, args.test_itemN)
item_num = 29419
args.user_num = user_num #22976
args.item_num = item_num #29419

# construct the train and test datasets
train_dataset = data_utils_v2.NCFData(
        train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils_v2.NCFData(
        test_data, item_num, train_mat, 0, False)

train_loader = data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=64)
test_loader = data.DataLoader(test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=64)

assert args.batch_size // len(args.gpu.split(",")) == args.batch_size / len(args.gpu.split(",")), 'Batch size is not divisible by num of gpus.'


########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
    assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
    assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(config.GMF_model_path)
    MLP_model = torch.load(config.MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

if config.model != "NCF-attn":
    model = model.NCF(user_num, item_num, 16, 3, args.dropout, config.model, GMF_model, MLP_model)
else:
    model = model.NCF_attn(item_num, user_num, args)

model = model.cuda()
model = torch.nn.DataParallel(model, device_ids=list(range(len(args.gpu.split(",")))))

loss_function = nn.CrossEntropyLoss()

if config.model == 'NeuMF-pre':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:#has to be further change to AdamW
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.1)

########################### TRAINING #####################################
#writer = SummaryWriter('./logs')

best_hr, best_ndcg, best_epoch = 0, 0, 0
print("Start training: select top_{} from {} items for recommendations".format(args.top_k, args.test_itemN))
for epoch in range(args.epochs):
    model.train() # Enable dropout (if have).
    start_time = time.time()
    print("Sampling negative samples ...")
    train_loader.dataset.ng_sample()
    
    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    print("Training ... lr:{}".format(get_learning_rate(optimizer)))

    losses = 0
    for i, data in enumerate(tqdm(train_loader)):
        user, item, label = data
        user = user.cuda()
        item = item.cuda()
        label = label.type(torch.LongTensor).cuda()
        #label = label.float().cuda()

        model.zero_grad()
        prediction = model(user, item)
        loss = loss_function(prediction, label)
        losses += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    #writer.add_scalar('Train/losses', losses, epoch)
    print("Loss: {}".format(losses))
    print("Evaluating ...")
    model.eval()
    HR, NDCG = evaluate_v2.metrics(model, test_loader, labels, args) #very slow, not parallel
    #writer.add_scalar('Train/HR', HR, epoch)
    #writer.add_scalar('Train/NDCG', NDCG, epoch)
    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
            time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if args.out:
            if not os.path.exists(config.model_path):
                os.mkdir(config.model_path)
            torch.save(model, 
                '{}{}_decLlayer{}_seqLen{}_hidDim{}_nHead{}_dropout{}_lr{}_Epoch{}_Batch{}_{}GPU.pth'.format(config.model_path, config.model, args.n_dec_layer, args.seq_len, args.hid_dim, args.nhead, args.dropout, args.lr, args.epochs, args.batch_size, len(args.gpu.split(",")))

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
                                    best_epoch, best_hr, best_ndcg))

fn = '{}{}_decLlayer{}_seqLen{}_hidDim{}_nHead{}_dropout{}_lr{}_Epoch{}_Batch{}_{}GPU.csv'.format(config.model_path, config.model, args.n_dec_layer, args.seq_len, args.hid_dim, args.nhead, args.dropout, args.lr, args.epochs, args.batch_size, len(args.gpu.split(",")))

with open(fn, 'w') as f:
    csv.writer(f).writerow(['model','n_dec_layer', 'seq_len', 'hid_dim', 'nhead', 'dropout', 'args.lr', 'epochs', 'best_epoch', 'batch_size', 'n_GPU', 'Recall@50', 'NDCG@50'])
    csv.writer(f).writerow([config.model, args.n_dec_layer, args.seq_len, args.hid_dim, args.nhead, args.dropout, args.lr, args.epochs, best_epoch, args.batch_size, len(args.gpu.split(",")), best_hr, best_ndcg])

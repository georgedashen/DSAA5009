'''
This version is tailor for taobao dataset
'''

import numpy as np
import torch
from torch.nn.functional import softmax
from tqdm import tqdm


def hit(gt_item, pred_items):
    hit = 0
    for item in gt_item:
        if item in pred_items: #how about len>1
            hit += 1
    return hit/len(gt_item)

# the same as HR
def recall(gt_item, pred_item):
    hit = 0
    for item in gt_item:
        if item in pred_items:
            hit += 1
    return hit/len(gt_item)

def ndcg(gt_item, pred_items):
    dcg = 0
    idcg = 0
    for i, p in enumerate(pred_items):
        if p in gt_item:
            dcg += 1 / np.log2(i + 2)
        if i < len(gt_item):
            idcg += 1 / np.log2(i + 2)
    ndcg = dcg / idcg
    return ndcg


def metrics(model, test_loader, labels, args):
    HR, NDCG = [], []
    predictions = []
    items = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            user, item, label = data
            user = user.cuda()
            item = item.cuda()

            prediction = model(user, item) #Batch * 2
            prediction = softmax(prediction, dim=1)[:,1]
            predictions.append(prediction)
            items.append(item)

    predictions = torch.cat(predictions)
    items = torch.cat(items)

    for i in range(args.user_num):
        _, indices = torch.topk(predictions[i*args.test_itemN:(i+1)*args.test_itemN], args.top_k)
        recommends = torch.take(items[i*args.test_itemN:(i+1)*args.test_itemN], indices).cpu().numpy().tolist()
        idx = labels[i*args.test_itemN:(i+1)*args.test_itemN] # 1 for pos, 0 for neg
        gt_item = [t.item() for t in items[i*args.test_itemN:(i+1)*args.test_itemN]]
        gt_item = np.array(gt_item)[[i==1 for i in idx]]
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

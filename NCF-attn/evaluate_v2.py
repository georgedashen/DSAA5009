import numpy as np
import torch
from torch.nn.functional import softmax
from tqdm import tqdm


def hit(gt_item, pred_items):
    if gt_item in pred_items: #how about len>1
        return 1
    return 0

def recall(gt_item, pred_item):
    pass


def ndcg(gt_item, pred_items):
    if gt_item in pred_items: #how about len>1
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0


def metrics(model, test_loader, args):
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
        gt_item = items[i*args.test_itemN].item() #could be len>1
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

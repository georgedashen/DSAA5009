import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='user-based', choices=['user-based', 'random', 'majority'])
parser.add_argument('--similarity', type=str, default='cosine', choices=['cosine', 'jacard'])
parser.add_argument('--k_user', type=int, default=5, help='how many similar users are chosen')
parser.add_argument('--top_n', type=int, default=1, help='how many items are recommanded')
parser.add_argument('--negative', type=str, default=None, help='Provide negative sampling')
parser.add_argument('--output', type=str, default='results', help='output dir')
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()

# 用户-物品交互矩阵，键为用户ID，值为该用户有过交互的物品ID列表
user_item_dict = {
    'user1': ['item1', 'item2', 'item3', 'item4'],
    'user2': ['item1', 'item2', 'item4'],
    'user3': ['item2', 'item3', 'item4'],
    'user4': ['item1', 'item3', 'item4'],
    'user5': ['item1', 'item2', 'item3'],
    'user6': ['item2', 'item4'],
    'user7': ['item1', 'item3'],
    'user8': ['item2', 'item3']
}

def jaccard_similarity(items1, items2):
    """
    :param set_a: 集合a
    :param set_b: 集合b
    :return: Jaccard相似度
    """
    intersection_size = len(set(items1) & set(items2))
    union_size = len(set(items1) | set(items1))
    return intersection_size / union_size

def cosine_sim_items(items1, items2):
    items_both = list(set(items1) | set(items2))
    if len(items_both) > 0:
        vec1 = [1 if item in items1 else 0 for item in items_both]
        vec2 = [1 if item in items2 else 0 for item in items_both]
        return cosine_similarity([vec1], [vec2])[0][0]

def cosine_sim_matrix(user1, user2):
    return cosine_similarity([user_items_matrix[user1]], [user_items_matrix[user2]])[0][0]


# 物品-用户交互矩阵，键为物品ID，值为有过交互该物品的用户ID列表
item_user_dict = {}
for user, items in user_item_dict.items():
    for item in items:
        if item not in item_user_dict:
            item_user_dict[item] = [user]
        else:
            item_user_dict[item].append(user)

# 构建用户-用户相似度矩阵
similarity_matrix = np.zeros((len(user_item_dict), len(user_item_dict)))
for i, (user1, items1) in enumerate(user_item_dict.items()):
    for j, (user2, items2) in enumerate(user_item_dict.items()):
        if i != j:
            # 计算余弦（或其他相似度）相似度
            similarity_matrix[i][j] = cosine_sim_items(items1, items2)

# 根据用户-用户相似度矩阵和测试集生成推荐结果
def user_based_recommendation(test_set, train_set, k_user=5, top_n=1, negative=None):
    """
    :param test_set: 字典结构，键为用户ID，值为该用户的测试集（未交互过的物品ID列表）
    :param train_set: 字典结构，键为用户ID，值为该用户的训练集（已知的物品ID列表）
    :param top_n: 推荐物品数
    :return: 字典结构，键为用户ID，值为该用户的推荐物品ID列表
    """
    if negative:
        negative_dict = {}
        with open(negative, 'r') as fd:
            line = fd.readline().rstrip()
            while line != None and line != '':
                arr = line.split('\t')
                u = eval(arr[0])[0]
                if u not in negative_dict.keys():
                    negative_dict[u] = []
                for i in range(len(eval(arr[0]))-1):
                    negative_dict[u].append(eval(arr[0])[i+1])
                for i in arr[1:]:
                    negative_dict[u].append(int(i))
                if len(negative_dict[u])>500:
                    print(f"Something wrong at user {u}")
                    break
                line = fd.readline().rstrip()
    
    recommendations = {}
    for i, user in enumerate(tqdm(test_set.keys())):
        # 找出与该用户最相似的K个用户
        k_similarities_indices = similarity_matrix[user].argsort()[::-1][:k_user]
        k_similarities = user_items_matrix[k_similarities_indices]

        # 计算推荐物品得分
        # 计算推荐物品得分
        # 根据得分对物品排序
        # 根据这5个人都有购买的东西推荐，同时考虑该物品的总购买人数
        k_freq = np.sum(k_similarities, 0)

        # 获取前N个推荐物品
        # if there is negative sampling, only recommended from give items
        # given_idx for each user
        labels = np.array([i for i in range(len(k_freq))]) # label for item
        if negative:
            given_idx = negative_dict[user]
        else:
            given_idx = [i for i in range(len(k_freq))] # all items
        labels = labels[given_idx]
        top_items_candidate = k_freq[given_idx].argsort()[::-1][:100] #maximum num. of bought item by user are 76
        top_items_candidate = labels[top_items_candidate] # get correct item id
        top_items=[]
        for item in top_items_candidate:
            if len(top_items) == top_n:
                break
            if item not in train_set[user]:
                top_items.append(item)
    
        recommendations[user] = top_items
    
    return recommendations

import random

def random_recommendation(test_set, train_set, top_n=1, negative=None):
    """
    :param train_set: 字典结构，键为用户ID，值为该用户的训练集（已知的物品ID列表）
    :param test_set: 字典结构，键为用户ID，值为该用户的测试集（未交互过的物品ID列表）
    :param top_n: 推荐物品数
    :return: 字典结构，键为用户ID，值为该用户的推荐物品ID列表
    """
    if negative:
        negative_dict = {}
        with open(negative, 'r') as fd:
            line = fd.readline().rstrip()
            while line != None and line != '':
                arr = line.split('\t')
                u = eval(arr[0])[0]
                if u not in negative_dict.keys():
                    negative_dict[u] = []
                for i in range(len(eval(arr[0]))-1):
                    negative_dict[u].append(eval(arr[0])[i+1])
                for i in arr[1:]:
                    negative_dict[u].append(int(i))
                if len(negative_dict[u])>500:
                    print(f"Something wrong at user {u}")
                    break
                line = fd.readline().rstrip()
        print("Negative_dict constructed!")
    
    recommendations = {}
    for i, user in enumerate(tqdm(test_set.keys())):
        # 从所有未交互过的物品中随机选择top_n个物品作为推荐结果
        while True:
            if negative:
                recommendations[user] = np.random.choice(negative_dict[user], top_n)
            else:
                recommendations[user] = np.random.choice(user_items_matrix.shape[1], top_n)
            if len(set(recommendations[user]) & set(train_set[user]))==0: #recommanded items have not benn bought yet
                break

    return recommendations

def majority_recommendation(test_set, train_set, top_n=1, negative=None):
    """
    :param train_set: 字典结构，键为用户ID，值为该用户的训练集（已知的物品ID列表）
    :param test_set: 字典结构，键为用户ID，值为该用户的测试集（未交互过的物品ID列表）
    :param top_n: 推荐物品数
    :return: 字典结构，键为用户ID，值为该用户的推荐物品ID列表
    """
    if negative:
        negative_dict = {}
        with open(negative, 'r') as fd:
            line = fd.readline().rstrip()
            while line != None and line != '':
                arr = line.split('\t')
                u = eval(arr[0])[0]
                if u not in negative_dict.keys():
                    negative_dict[u] = []
                for i in range(len(eval(arr[0]))-1):
                    negative_dict[u].append(eval(arr[0])[i+1])
                for i in arr[1:]:
                    negative_dict[u].append(int(i))
                if len(negative_dict[u])>500:
                    print(f"Something wrong at user {u}")
                    break
                line = fd.readline().rstrip()
        print("Negative_dict constructed!")
    
    recommendations = {}
    for i, user in enumerate(tqdm(test_set.keys())):
        if negative:
            labels = np.array([i for i in range(len(item_freq))])
            labels = labels[negative_dict[user]]
            recommendations[user] = labels[item_freq[[negative_dict[user]]].argsort()[::-1][:top_n]]
        else:
            top_items=[]
            items = item_freq.argsort()[::-1][:200]
            for item in items:
                if item not in train_set[user]:
                    top_items.append(item)
                if len(top_items) == top_n:
                    break
            recommendations[user] = top_items

    return recommendations

def recall_at_k(test_set, recommendations, k=args.top_n):
    """
    :param test_set: 字典结构，键为用户ID，值为该用户的测试集（未交互过的物品ID列表）
    :param recommendations: 字典结构，键为用户ID，值为该用户的推荐物品ID列表
    :param k: 推荐物品数
    :return: Recall@k
    """
    recall = 0
    for user, test_items in test_set.items():
        num_hits = 0
        for item in test_items:
            if item in recommendations[user][:k]:
                num_hits += 1
        recall += num_hits/len(test_items)
    return recall / len(test_set)


def ndcg_at_k(test_set, recommendations, k=args.top_n):
    """
    :param test_set: 字典结构，键为用户ID，值为该用户的测试集（未交互过的物品ID列表）
    :param recommendations: 字典结构，键为用户ID，值为该用户的推荐物品ID列表
    :param k: 推荐物品数
    :return: NDCG@k
    """
    ndcg = 0
    for user, test_items in test_set.items():
        predicted_items = recommendations[user][:k]
        if len(predicted_items) > 0:
            dcg = 0
            idcg = 0
            for i, item in enumerate(predicted_items):
                if item in test_items:
                    dcg += 1 / np.log2(i + 2)
                if i < len(test_items):
                    idcg += 1 / np.log2(i + 2)
            ndcg += dcg / idcg
    return ndcg / len(test_set)

'''
这里我假设您已经将以上推荐算法和数据集集成到一个完整的Python程序中，然后您可以使用这些函数来计算Recall@k和NDCG@k指标。对于Recall@k，您可以调用recall_at_k(test_set, recommendations, k)函数，其中test_set和recommendations参数与您的程序中的数据结构相同。对于NDCG@k，您可以调用ndcg_at_k(test_set, recommendations, k)函数，其中test_set和recommendations参数与您的程序中的数据结构相同。这些函数都会返回相应的指标值。
'''

if __name__ == '__main__':

    import os
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    random.seed(123)
    np.random.seed(123)

    print('Loading dataset ...')
    import pickle
    with open('train_user_consumed.pkl', 'rb') as pkl:
        train = pickle.load(pkl)
    
    with open('val_user_consumed.pkl', 'rb') as pkl:
        val = pickle.load(pkl)
    
    with open('test_user_consumed.pkl', 'rb') as pkl:
        test = pickle.load(pkl)
    
    #print(train)
    #print(val)
    #print(test)
    
    # 构建所有物品列表
    all_items = []
    for i, (user, items) in enumerate(train.items()):
        all_items += items
    for i, (user, items) in enumerate(val.items()):
        all_items += items
    for i, (user, items) in enumerate(test.items()):
        all_items += items

    all_items = set(all_items)
    
    print('Creating user-item matrix ...')
    #构建用户-物品交互矩阵
    user_items_matrix = np.zeros((len(train), max(all_items)+1))
    for i, (user, items) in enumerate(train.items()):
        for item in items:
            user_items_matrix[int(user)][item] = 1

    item_freq = np.sum(user_items_matrix, 0)

    from tqdm import tqdm
    
    print('Generating recommendation ...')
   
    if args.test:
        val = test
    if args.model == 'user-based':
        print('Calculating similarity ...')
        # 构建用户-用户相似度矩阵
        similarity_matrix = np.zeros((len(train), len(train)))
        if args.similarity == 'cosine':
            similarity_matrix = cosine_similarity(user_items_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        
        rec = user_based_recommendation(val, train, k_user=args.k_user, top_n=args.top_n, negative=args.negative)
    
    if args.model == 'random':
        rec = random_recommendation(val, train, top_n=args.top_n, negative=args.negative)
    
    if args.model == 'majority':
        rec = majority_recommendation(val, train, top_n=args.top_n, negative=args.negative)

    #print(rec)
    print(f'recall@k: {recall_at_k(val, rec)}')
    print(f'ndcg@k: {ndcg_at_k(val, rec)}')

    print('Done!')

    stats = []
    stats.append(
        {
            'recall@k': recall_at_k(val, rec),
            'ndcg@k': ndcg_at_k(val, rec),
            'model': args.model,
            'k_user': args.k_user,
            'top_n': args.top_n
        }
    )

    import pandas as pd
    df_stats = pd.DataFrame(data=stats)
    filename = args.output + '/' + 'stat_' + args.model + '_k' + str(args.k_user) + '_n' + str(args.top_n) + '.csv'
    df_stats.to_csv(filename, index=False)


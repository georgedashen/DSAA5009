import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
def user_based_recommendation(test_set, k_user=5, top_n=1):
    """
    :param test_set: 字典结构，键为用户ID，值为该用户的测试集（未交互过的物品ID列表）
    :param top_n: 推荐物品数
    :return: 字典结构，键为用户ID，值为该用户的推荐物品ID列表
    """
    recommendations = {}
    for user in test_set.keys():
        # 找出与该用户最相似的K个用户
        k_similarities_indices = similarity_matrix[user].argsort()[::-1][:k_user]
        k_similarities = user_items_matrix[k_similarities_indices]

        # 计算推荐物品得分
        # 计算推荐物品得分
        # 根据得分对物品排序
        # 根据这5个人都有购买的东西推荐，同时考虑该物品的总购买人数
        k_freq = np.sum(k_similarities, 0)

        # 获取前N个推荐物品
        top_items = k_freq.argsort()[::-1][:top_n]
    
        recommendations[user] = top_items
    
    return recommendations

import random

def random_recommendation(test_set, top_n=1):
    """
    :param test_set: 字典结构，键为用户ID，值为该用户的测试集（未交互过的物品ID列表）
    :param top_n: 推荐物品数
    :return: 字典结构，键为用户ID，值为该用户的推荐物品ID列表
    """
    recommendations = {}
    for user in test_set.keys():
        # 从所有未交互过的物品中随机选择top_n个物品作为推荐结果
        recommendations[user] = np.random.choice(user_items_matrix.shape[1], top_n)
    return recommendations


def recall_at_k(test_set, recommendations, k=1):
    """
    :param test_set: 字典结构，键为用户ID，值为该用户的测试集（未交互过的物品ID列表）
    :param recommendations: 字典结构，键为用户ID，值为该用户的推荐物品ID列表
    :param k: 推荐物品数
    :return: Recall@k
    """
    num_hits = 0
    num_interactions = 0
    for user, test_items in test_set.items():
        for item in test_items:
            if item in recommendations[user][:k]:
                num_hits += 1
        num_interactions += len(test_items)
    return num_hits / num_interactions


def ndcg_at_k(test_set, recommendations, k=1):
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

    all_items = set(all_items)
        
        #构建用户-物品交互矩阵
    user_items_matrix = np.zeros((len(train), max(all_items)+1))
    for i, (user, items) in enumerate(train.items()):
        for item in items:
            user_items_matrix[int(user)][item] = 1

    item_freq = np.sum(user_items_matrix, 0)

    from tqdm import tqdm
    # 构建用户-用户相似度矩阵
    similarity_matrix = np.zeros((len(train), len(train)))
    similarity_matrix = cosine_similarity(user_items_matrix)
    '''
    for i in tqdm(range(len(train)-1)):
        for j in range(i+1, len(train)):
            # 计算余弦（或其他相似度）相似度
            similarity_matrix[i][j] = cosine_sim_matrix(i, j)
    '''
    np.fill_diagonal(similarity_matrix, 0)

    rec = user_based_recommendation(val)

    #print(rec)
    recall_at_k(val, rec)
    ndcg_at_k(val, rec)


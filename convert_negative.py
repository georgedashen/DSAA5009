import numpy as np
from tqdm import tqdm
import pickle
import random

N = 500
# N = 29148 for all items
random.seed(123)

# load train
with open('train_user_consumed.pkl', 'rb') as f:
    train = pickle.load(f)

# construct train user-item matrix
user_items_matrix = np.zeros((22976, 29149))
for i, (user, items) in enumerate(train.items()):
    for item in items:
        user_items_matrix[int(user)][item] = 1

#load val
with open('val_user_consumed.pkl', 'rb') as f:
    data = pickle.load(f)

#val_negative: generate N items including both pos and neg samples
with open(f'val_negative_{N}', 'w') as f:
    for i, (x, y_list) in enumerate(tqdm(data.items())):
        y_str = ','.join(str(y) for y in y_list) # positive samples
        N_rest = N - len(y_list) # number of negative samples
        values = [i for i in range(29149) if i not in y_list and user_items_matrix[x][i]==0]
        random.shuffle(values) # randomized
        values = values[:N_rest] #negative sampling
        for value in values:
            user_items_matrix[x][value] = 1 # update user-item matrix
        values_str = '\t'.join(str(i) for i in values)
        line = f"({x},{y_str})\t{values_str}\n"
        f.write(line)

#load val
with open('test_user_consumed.pkl', 'rb') as f:
    data = pickle.load(f)
    
#test_negative: generate N items including both pos and neg samples
with open(f'test_negative_{N}', 'w') as f:
    for i, (x, y_list) in enumerate(tqdm(data.items())):
        y_str = ','.join(str(y) for y in y_list) # positive samples
        N_rest = N - len(y_list) # number of negative samples
        values = [i for i in range(29149) if i not in y_list and user_items_matrix[x][i]==0]
        random.shuffle(values) # randomized
        values = values[:N_rest] #negative sampling
        values_str = '\t'.join(str(i) for i in values)
        line = f"({x},{y_str})\t{values_str}\n"
        f.write(line)

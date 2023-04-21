# dataset name 
dataset = 'ml-1m'
# assert dataset in ['ml-1m', 'pinterest-20']

# model name 
#model = 'NCF-attn'
model = 'MLP'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre', 'NCF-attn']

# paths
main_path = './Data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = './test/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'

import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformer import build_decoder
import math


class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                    dropout, model, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """        
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
                user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
                item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num 
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                    a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(
                            self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                            self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                            self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                            self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight, 
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                        self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

class NCF_attn(nn.Module):
    def __init__(self, item_num, user_num, args):
        super(NCF_attn, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        hid_num: latent vector dimension for items and users;
        n_dec_layer: the number of decoder layer;
        dropout: dropout rate between fully connected layers;
        """
        self.dropout = args.dropout
        self.embed_item = nn.Embedding(item_num, args.hid_dim*args.seq_len)
        self.embed_user = nn.Embedding(user_num, args.hid_dim*args.seq_len)
        self.decoder = build_decoder(args)
        #self.fc = GroupWiseLinear(2, args.fc_dim*args.seq_len, bias=True) # Y/N ?
        self.fc = FC_Decoder(2, args.hid_dim*args.seq_len, args.fc_dim, args.dropout, args.activation)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        
    def forward(self, user, item):
        item_embedding = self.embed_item(item)
        user_embedding = self.embed_user(user) #pre-model
        output = self.decoder(user_embedding, item_embedding, pos_embed=None)
        output = self.fc(output)

        return output


class FC_Decoder(nn.Module):
    def __init__(self, num_class, attn_dim, fc_dim, dropout, activation):
        super().__init__()
        self.num_class = num_class

        self.output_layer1 = nn.Linear(attn_dim, fc_dim)
        self.activation = _get_activation_fn(activation)
        self.dropout = torch.nn.Dropout(dropout)

        self.output_layer2 = nn.Linear(fc_dim, num_class)

    def forward(self, hs):
        hs = hs.reshape(hs.shape[0],-1) #Batch, Len*Dim

        hs = self.output_layer1(hs) #Batch, fc_dim
        hs = self.activation(hs)
        hs = self.dropout(hs)

        out = self.output_layer2(hs) #Batch, 2
        return out



# no need to use it if not multi-label task
class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, attn_dim, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W1 = nn.Parameter(torch.Tensor(1, hidden_dim, attn_dim))
        self.W2 = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b1 = nn.Parameter(torch.Tensor(1, num_class))
            self.b2 = nn.Parameter(torch.Tensor(1, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.hidden_dim):
            self.W1[0][i].data.uniform_(-stdv, stdv)
        for i in range(self.num_class):
            self.W2[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.hidden_dim):
                self.b1[0][i].data.uniform_(-stdv, stdv)
            for i in range(self.num_class):
                self.b2[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,L,D
        x = (self.W1 * x).sum(-1)
        if self.bias:
            x = x + self.b1
        x = (self.W2 * x).sum(-1)
        if self.bias:
            x = x + self.b2
        
        return x



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


import sys
import os
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autogcrad
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
from tqdm import tqdm, trange
import collections
from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
import pickle
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer


s = input("sentences: ")
from pytorch_pretrained_bert.tokenization import BertTokenizer
bert_model_scale = 'bert-base-cased'
do_lower_case = False
tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)
token = tokenizer.tokenize(s)

label_type = ['X', '[CLS]', '[SEP]', 'O', 'L', 'BL', 'EL']
label_map = {}

for (i, label) in enumerate(label_type):
    
    # Map it to its integer
    label_map[label] = i

add_label = 'X'
tokens = ['[CLS]']
predict_mask = [0]
label_ids = [label_map['[CLS]']]

input_ids = tokenizer.convert_tokens_to_ids(token)

segment_ids = [0] * len(input_ids)
input_mask = [1] * len(input_ids)

class BERT_CRF_NER(nn.Module):

    def __init__(self, bert_model, start_label_id, stop_label_id, num_labels, max_seq_length, batch_size, device):
        super(BERT_CRF_NER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        # self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device=device

        # use pretrainded BertModel
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.2)
        # Maps the output of the bert into label space.
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)
        # self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0

        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_bert_features(self, input_ids, segment_ids, input_mask):
        '''
        sentances -> word embedding -> lstm -> MLP -> feats
        '''
        bert_seq_out, _ = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2label(bert_seq_out)
        return bert_feats

    def _score_sentence(self, feats, label_ids):
        '''
        Gives the score of a provided label sequence
        p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size,self.num_labels,self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0],1)).to(device)
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t]*self.num_labels+label_ids[:, t-1]).view(-1,1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1,1)).view(-1,1)
        return score

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.num_labels,self.num_labels)

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t+1].gather(-1,path[:, t+1].view(-1,1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        forward_score = self._forward_alg(bert_feats)
        # p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        gold_score = self._score_sentence(bert_feats, label_ids)
        # - log[ p(X=w1:t,Zt=tag1:t)/p(X=w1:t) ] = - log[ p(Zt=tag1:t|X=w1:t) ]
        return torch.mean(forward_score - gold_score)

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids, segment_ids, input_mask):
        # Get the emission scores from the BiLSTM
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)

        # Find the best path, given the features.
        score, label_seq_ids = self._viterbi_decode(bert_feats)
        return score, label_seq_ids
    
max_seq_length = 500
batch_size = 2 
cuda_yes = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_yes else "cpu")

bert_model = BertModel.from_pretrained(bert_model_scale)
model = BERT_CRF_NER(bert_model, label_map['[CLS]'], label_map['[SEP]'], len(label_map), max_seq_length, batch_size, device)

load_checkpoint = True
output_dir = '/home/ducnh/PycharmProjects/195/model/'

if load_checkpoint and os.path.exists(output_dir+'/ner_bert_crf_checkpoint.pt'):
    checkpoint = torch.load(output_dir+'/ner_bert_crf_checkpoint.pt', map_location='cpu')
    start_epoch = checkpoint['epoch']+1
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    pretrained_dict=checkpoint['model_state']
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    # print('Loaded the pretrain data model, epoch:',checkpoint['epoch'],'valid acc:',
    #         checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])


model.to(device)

predict_mask = []
for i in token:
    predict_mask.append(1)

input_ids = torch.tensor([input_ids]).cuda()
segment_ids = torch.tensor([segment_ids]).cuda()
input_mask = torch.tensor([input_mask]).cuda()

predict_mask = torch.tensor([predict_mask], device='cuda:0', dtype=torch.uint8)

_, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)

valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)

all_preds = []
all_preds.extend(valid_predicted.tolist())
all_preds

from transformers import BertTokenizer
tz = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

w = []
result = []
for i in range(len(all_preds)):
    if all_preds[i] == 5 or all_preds[i] == 4 or all_preds[i] == 6:
        w.append(token[i])
result.append(w)

sentences = []
for i in result:

    output_string = tz.convert_tokens_to_string(i)
    output_string = output_string.replace(" ", "")
    print(output_string)

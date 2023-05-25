
# %%
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

from pre_data import *
from evaluate import *
from bert_crf import *

# Set path
def set_work_dir(local_path="data", server_path="data"):
    path_base = "/home/ducnh/PycharmProjects/195"
    if (os.path.exists(path_base+'/'+local_path)):
        os.chdir(path_base+'/'+local_path)
    elif (os.path.exists(path_base+'/'+server_path)):
        os.chdir(path_base+'/'+server_path)
    else:
        raise Exception('Set work path error!')


def get_data_dir(local_path="data", server_path="data"):
    path_base = "/home/ducnh/PycharmProjects/195"
    if (os.path.exists(path_base+'/'+local_path)):
        return path_base+'/'+local_path
    elif (os.path.exists(path_base+'/'+server_path)):
        return path_base+'/'+server_path
    else:
        raise Exception('get data path error!')

# check version
print('Python version ', sys.version)
print('PyTorch version ', torch.__version__)
set_work_dir()
print('Current dir:', os.getcwd())
# check gpu
print('Cuda is available?', cuda_yes)
device = torch.device("cuda:0" if cuda_yes else "cpu")
print('Device:', device)

data_dir = os.path.join(get_data_dir())
do_train = True
do_eval = True
do_predict = True
load_checkpoint = True
# "The vocabulary file that the BERT model was trained on."
# setup tham số cho bert
max_seq_length = 500 #256
batch_size = 2 #32
# "The initial learning rate for Adam."
learning_rate0 = 5e-5
lr0_crf_fc = 8e-5
weight_decay_finetune = 1e-5 #0.01
weight_decay_crf_fc = 5e-6 #0.005
total_train_epochs = 3
gradient_accumulation_steps = 1
warmup_proportion = 0.1
output_dir = '/home/ducnh/PycharmProjects/195/model'
bert_model_scale = 'bert-base-cased'
do_lower_case = False
'''
Prepare data set
'''
# random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
if cuda_yes:
    torch.cuda.manual_seed_all(44)

# Load pre-trained model tokenizer (vocabulary)
conllProcessor = CoNLLDataProcessor()
label_list = conllProcessor.get_labels()
label_map = conllProcessor.get_label_map()
train_examples = conllProcessor.get_train_examples(data_dir)
dev_examples = conllProcessor.get_dev_examples(data_dir)
test_examples = conllProcessor.get_test_examples(data_dir)

total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)
# print(train_examples)
# print(total_train_steps)
print("***** Running training *****")
print("  Num examples = %d"% len(train_examples))
print("  Batch size = %d"% batch_size)
print("  Num steps = %d"% total_train_steps)
tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

train_dataset = NerDataset(train_examples,tokenizer,label_map,max_seq_length)
dev_dataset = NerDataset(dev_examples,tokenizer,label_map,max_seq_length)
test_dataset = NerDataset(test_examples,tokenizer,label_map,max_seq_length)

train_dataloader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=NerDataset.pad)

dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=NerDataset.pad)

test_dataloader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=NerDataset.pad)


#%%
# '''
# #####  Use only BertForTokenClassification  #####
# '''
print('*** Use only BertForTokenClassification ***')

if load_checkpoint and os.path.exists(output_dir+'/ner_bert_checkpoint.pt'):
    checkpoint = torch.load(output_dir+'/ner_bert_checkpoint.pt', map_location='cpu')
    start_epoch = checkpoint['epoch']+1
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    model = BertForTokenClassification.from_pretrained(
        bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list))
    print('Loaded the pretrain NER_BERT model, epoch:',checkpoint['epoch'],'valid acc:',
            checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
else:
    start_epoch = 0
    valid_acc_prev = 0
    valid_f1_prev = 0
    model = BertForTokenClassification.from_pretrained(
        bert_model_scale, num_labels=len(label_list))

model.to(device)

# Prepare optimizer
named_params = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate0, warmup=warmup_proportion, t_total=total_train_steps)


#%%
global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    train_start = time.time()
    model.train()
    optimizer.zero_grad()
    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = learning_rate0 * warmup_linear(global_step_th/total_train_steps, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1
        print("Epoch:{}-{}/{}, CrossEntropyLoss: {} ".format(epoch, step, len(train_dataloader), loss.item()))

    print('--------------------------------------------------------------')
    print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss, (time.time() - train_start) / 60.0))
    valid_acc, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')
    # Save a checkpoint
    if valid_f1 > valid_f1_prev:
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
            'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': do_lower_case},
                    os.path.join(output_dir, 'ner_bert_checkpoint.pt'))
        valid_f1_prev = valid_f1

evaluate(model, test_dataloader, batch_size, total_train_epochs-1, 'Test_set')
#%%
'''
Test_set prediction using the best epoch of NER_BERT model
'''
checkpoint = torch.load(output_dir+'/ner_bert_checkpoint.pt', map_location='cpu')
epoch = checkpoint['epoch']
valid_acc_prev = checkpoint['valid_acc']
valid_f1_prev = checkpoint['valid_f1']
model = BertForTokenClassification.from_pretrained(
    bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list))
# if os.path.exists(output_dir+'/ner_bert_crf_checkpoint.pt'):
model.to(device)
print('Loaded the pretrain NER_BERT model, epoch:',checkpoint['epoch'],'valid acc:',
        checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

model.to(device)
evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')
#%%
'''
#####  Use BertModel + CRF  #####
CRF is for transition and the maximum likelyhood estimate(MLE).
Bert is for latent label -> Emission of word embedding.
'''
print('*** Use BertModel + CRF ***')


start_label_id = conllProcessor.get_start_label_id()
stop_label_id = conllProcessor.get_stop_label_id()

bert_model = BertModel.from_pretrained(bert_model_scale)
model = BERT_CRF_NER(bert_model, start_label_id, stop_label_id, len(label_list), max_seq_length, batch_size, device)

#%%
if load_checkpoint and os.path.exists(output_dir+'/ner_bert_crf_checkpoint.pt'):
    checkpoint = torch.load(output_dir+'/ner_bert_crf_checkpoint.pt', map_location='cpu')
    start_epoch = checkpoint['epoch']+1
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    pretrained_dict=checkpoint['model_state']
    net_state_dict = model.stloadate_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    print('Loaded the pretrain data model, epoch:',checkpoint['epoch'],'valid acc:',
            checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
else:
    start_epoch = 0
    valid_acc_prev = 0
    valid_f1_prev = 0

model.to(device)

# Prepare optimizer
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
        and not any(nd in n for nd in new_param)], 'weight_decay': weight_decay_finetune},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
        and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n in ('transitions','hidden2label.weight')] \
        , 'lr':lr0_crf_fc, 'weight_decay': weight_decay_crf_fc},
    {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
        , 'lr':lr0_crf_fc, 'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate0, warmup=warmup_proportion, t_total=total_train_steps)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate0)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

#%%
# train procedure
global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)

# train_start=time.time()
# for epoch in trange(start_epoch, total_train_epochs, desc="Epoch"):
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    train_start = time.time()
    model.train()
    optimizer.zero_grad()
    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

        neg_log_likelihood = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)

        if gradient_accumulation_steps > 1:
            neg_log_likelihood = neg_log_likelihood / gradient_accumulation_steps

        neg_log_likelihood.backward()

        tr_loss += neg_log_likelihood.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = learning_rate0 * warmup_linear(global_step_th/total_train_steps, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1

        print("Epoch:{}-{}/{}, Negative loglikelihood: {} ".format(epoch, step, len(train_dataloader), neg_log_likelihood.item()))

    print('--------------------------------------------------------------')
    print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss, (time.time() - train_start)/60.0))
    valid_acc, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')

    # Save a checkpoint
    if valid_f1 > valid_f1_prev:
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
            'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': do_lower_case},
                    os.path.join(output_dir, 'ner_bert_crf_checkpoint.pt'))
        valid_f1_prev = valid_f1

evaluate(model, test_dataloader, batch_size, total_train_epochs-1, 'Test_set')
#%%
'''
Test_set prediction using the best epoch of data model
'''
checkpoint = torch.load(output_dir+'/ner_bert_crf_checkpoint.pt', map_location='cpu')
epoch = checkpoint['epoch']
valid_acc_prev = checkpoint['valid_acc']
valid_f1_prev = checkpoint['valid_f1']
pretrained_dict=checkpoint['model_state']
net_state_dict = model.state_dict()
pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
net_state_dict.update(pretrained_dict_selected)
model.load_state_dict(net_state_dict)
print('Loaded the pretrain  data  model, epoch:',checkpoint['epoch'],'valid acc:',
      checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

model.to(device)
evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')

#%%
model.eval()
with torch.no_grad():
    demon_dataloader = data.DataLoader(dataset=test_dataset,
                                batch_size=10,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=NerDataset.pad)
    for batch in demon_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
        # _, predicted = torch.max(out_scores, -1)
        valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
        # valid_label_ids = torch.masked_select(label_ids, predict_mask)
        for i in range(10):
            print(predicted_label_seq_ids[i])
            print(label_ids[i])
            new_ids=predicted_label_seq_ids[i].cpu().numpy()[predict_mask[i].cpu().numpy()==1]
            print(list(map(lambda i: label_list[i], new_ids)))
            print(test_examples[i].labels)
        break
#%%
print(conllProcessor.get_label_map())


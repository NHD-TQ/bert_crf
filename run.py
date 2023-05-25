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
from bert_crf import *

s = input("sentences: ")
from pytorch_pretrained_bert.tokenization import BertTokenizer
bert_model_scale = 'bert-base-cased'
do_lower_case = False
tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)
token = tokenizer.tokenize(s)
# print(token)

def write_word_list(word_list, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for i, word in enumerate(word_list):
            line = f"{word} O O L\n"
            file.write(line)

write_word_list(token, "input.txt")


max_seq_length = 500
class InputExample(object):
    def __init__(self, guid, words, labels):

        self.guid = guid
        # list of words of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.words = words
        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.labels = labels

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        with open(input_file) as f:
            # out_lines = []
            out_lists = []
            entries = f.read().strip().split("\n\n")
            for entry in entries:
                words = []
                ner_labels = []
                pos_tags = []
                bio_pos_tags = []
                for line in entry.splitlines():
                    pieces = line.strip().split()
                    if len(pieces) < 1:
                        continue
                    word = pieces[0]
                    # if word == "-DOCSTART-" or word == '':

                    words.append(word)
                    pos_tags.append(pieces[1])
                    bio_pos_tags.append(pieces[2])
                    ner_labels.append(pieces[-1])

                out_lists.append([words,pos_tags,bio_pos_tags,ner_labels])
        return out_lists
class CoNLLDataProcessor(DataProcessor):
    '''
    CoNLL-2003
    '''

    def __init__(self):
        self._label_types = ['X', '[CLS]', '[SEP]', 'O', 'L', 'BL', 'EL']
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i,
                           label in enumerate(self._label_types)}

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "input.txt")))

    def get_labels(self):
        return self._label_types

    def get_num_labels(self):
        return self.get_num_labels

    def get_label_map(self):
        return self._label_map

    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def _create_examples(self, all_lists):
        examples = []
        for (i, one_lists) in enumerate(all_lists):
            guid = i
            words = one_lists[0]
            labels = one_lists[-1]
            examples.append(InputExample(
                guid=guid, words=words, labels=labels))
        return examples

    def _create_examples2(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text = line[0]
            ner_label = line[-1]
            examples.append(InputExample(
                guid=guid, text_a=text, labels_a=ner_label))
        return examples

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids,  predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids

def example2feature(example, tokenizer, label_map, max_seq_length):

    add_label = 'X'
    # tokenize_count = []
    tokens = ['[CLS]']
    predict_mask = [0]
    label_ids = [label_map['[CLS]']]
    for i, w in enumerate(example.words):
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        # tokenize_count.append(len(sub_words))
        tokens.extend(sub_words)
        for j in range(len(sub_words)):
            if j == 0:
                predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
            else:
                # '##xxx' -> 'X' (see bert paper)
                predict_mask.append(0)
                label_ids.append(label_map[add_label])

    # truncate
    if len(tokens) > max_seq_length - 1:
        print('Example No.{} is too long, length is {}, truncated to {}!'.format(example.guid, len(tokens), max_seq_length))
        tokens = tokens[0:(max_seq_length - 1)]
        predict_mask = predict_mask[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]
    tokens.append('[SEP]')
    predict_mask.append(0)
    label_ids.append(label_map['[SEP]'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    feat=InputFeatures(
                # guid=example.guid,
                # tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                predict_mask=predict_mask,
                label_ids=label_ids)

    return feat

class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.examples=examples
        self.tokenizer=tokenizer
        self.label_map=label_map
        self.max_seq_length=max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat=example2feature(self.examples[idx], self.tokenizer, self.label_map, max_seq_length)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids

    @classmethod
    def pad(cls, batch):

        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: X for padding
        input_ids_list = torch.LongTensor(f(0, maxlen))
        input_mask_list = torch.LongTensor(f(1, maxlen))
        segment_ids_list = torch.LongTensor(f(2, maxlen))
        predict_mask_list = torch.ByteTensor(f(3, maxlen))
        label_ids_list = torch.LongTensor(f(4, maxlen))

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list
    
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
output_dir = '/home/195_main/195/model'
bert_model_scale = 'bert-base-cased'
do_lower_case = False

conllProcessor = CoNLLDataProcessor()
label_list = conllProcessor.get_labels()
label_map = conllProcessor.get_label_map()


load_checkpoint = True
bert_model = BertModel.from_pretrained(bert_model_scale)
start_label_id = conllProcessor.get_start_label_id()
stop_label_id = conllProcessor.get_stop_label_id()
model = BERT_CRF_NER(bert_model, start_label_id, stop_label_id, len(label_list), max_seq_length, batch_size, device)

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
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    # print('Loaded the pretrain data model, epoch:',checkpoint['epoch'],'valid acc:',
    #         checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
else:
    start_epoch = 0
    valid_acc_prev = 0
    valid_f1_prev = 0

model.to(device)

test_examples = conllProcessor.get_test_examples('/home/195_main/195/')

tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

test_dataset = NerDataset(test_examples,tokenizer,label_map,max_seq_length)

test_dataloader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=NerDataset.pad)

predict_dataloader = test_dataloader

model.eval()
all_preds = []
all_labels = []
total=0
correct=0
start = time.time()
with torch.no_grad():
    for batch in predict_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
        valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
        valid_label_ids = torch.masked_select(label_ids, predict_mask)
        all_preds.extend(valid_predicted.tolist())
        all_labels.extend(valid_label_ids.tolist())
        # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
        total += len(valid_label_ids)
        correct += valid_predicted.eq(valid_label_ids).sum().item()

from transformers import BertTokenizer
tz = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

import pandas as pd

def read_text_to_dataframe(file_path):
    with open(file_path, mode='r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        if line.strip() == "":
            data.append([""] * 5)  
        else:
            # Tách cột dữ liệu theo khoảng trắng
            row = line.strip().split()
            if len(row) < 5:
                row.extend([""] * (5 - len(row)))  # Đảm bảo mỗi dòng có đúng 5 cột
            data.append(row)

    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame(data, columns=['Text', 'a', 'j', 'b', 'k'])

    return df
file_path = 'input.txt'
# Đọc từ file txt vào DataFrame
df = read_text_to_dataframe(file_path)
a = all_preds
for i in range(len(df['Text'])):
    if df['Text'][i] == "":
        a.insert(i, 7)

link = []
while True:
    w = []
    for i in range(len(a)):
        if a[i] != 7:
            if a[i] == 5 or a[i] == 4 or a[i] == 6:
                w.append(df['Text'][i])
        else:
            w = []
    link.append(w)
    break
sentences = []
for i in link:
    output_string = tz.convert_tokens_to_string(i)
    output_string = output_string.replace(" ", "")
    sentences.append(output_string)

print("Result: ", end='')
for i in sentences:
    print(i)
  
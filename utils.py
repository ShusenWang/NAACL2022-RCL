import math
import numpy as np
import time
import datetime
import os
import csv
import json
import random
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))

def read_json(data_file):
    with open(data_file,'r') as r:
        data = json.load(r)
    return data


def read_data(data, labellist, use_mark=True):
    sentences = [] 
    labels = [] 
    label_ids = [] 
    label2id = {} 
    with open(data, 'r', encoding='utf-8')as f:  
        fieldnames = ['sentences', 'label']
        data = csv.DictReader(f, fieldnames=fieldnames)
        for id, row in enumerate(data):
            if id == 0:
                continue
            # id += 1
            elif row['label'] in labellist:
                if use_mark:
                    sentences.append(row['sentences'])
                else:
                    sentences.append(row['sentences'].replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '').replace('  ', ' '))
                labels.append(row['label'])
                
    for label in set(labels):
        label2id[label] = len(label2id)
    label_ids = [label2id[i] for i in labels]
    return sentences, labels, label_ids, label2id


def get_contrastive_data(sentences, labels=None):
    contrastive_sentences = []
    contrastive_labels = []
    for sentence in sentences:
        sentence_pair = [sentence, sentence]
        contrastive_sentences.append(sentence_pair)
    
    if labels:
        contrastive_labels = labels
    
    return contrastive_sentences, contrastive_labels

def get_contrastive_feature(sentence_pairs, tokenizer, max_length):
    """
    sentence_pairs: str 
    labels: int 
    """
    contrastive_features = []
    for sentence_pair in sentence_pairs:
        sent_feature = tokenizer(
            sentence_pair,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )
        contrastive_features.append(sent_feature)
    return contrastive_features

def get_entity_idx(contrastive_features, tokenizer):  
    e1_tks_id = tokenizer.convert_tokens_to_ids('<e1>')
    e2_tks_id = tokenizer.convert_tokens_to_ids('<e2>')
    entity_idx = []
    for contrastive_feature in contrastive_features :
        en_idx = []
        for input_id in contrastive_feature['input_ids']:
            e1_idx = (input_id == e1_tks_id).nonzero().flatten().tolist()[0] 
            e2_idx = (input_id == e2_tks_id).nonzero().flatten().tolist()[0]  
            en_idx.append((e1_idx, e2_idx))
        entity_idx.append(en_idx)
    return entity_idx


def get_labellist(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        labellist = set()
        fieldnames = ['sentences', 'label']
        data = csv.DictReader(f, fieldnames=fieldnames)
        for id, row in enumerate(data):
            if id == 0:
                continue
            labellist.add(row['label'])
        if 'Other' in labellist:
            labellist.remove('Other')
        labellist = list(labellist)
    return labellist







import argparse
import math
import numpy as np
import time
import datetime
import os
import csv
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig

from utils import set_seed, format_time, read_json, read_data, get_contrastive_data, get_contrastive_feature, get_entity_idx, get_labellist
from dataset import RCLdataset, collate_fn
from model import RCL
from evaluation import ClusterEvaluation, standard_metric_result
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score



def train(contrastive_dataset, model, device, train_batch_size, train_epochs, seeds, save_model, collate_fn):

    train_sampler = RandomSampler(contrastive_dataset)
    train_dataloader = DataLoader(contrastive_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=collate_fn)
    t_total = len(train_dataloader) * train_epochs 
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    # Train
    model.to(device)
    model.zero_grad()
    set_seed(seeds) 
    
    training_stats = []
    global_step = 0
    best_eval_loss = 10
    # 统计整个训练时长
    total_t0 = time.time()
    for i in range(train_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(i + 1, train_epochs))
        print('Training...')
        total_train_loss = 0
        t0 = time.time() 
        
        for step, data in enumerate(train_dataloader):
            model.train()
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            entity_idx = data['entity_idx'].to(device)
            if 'token_type_ids' in data.keys():
                token_type_ids = data['token_type_ids'].to(device)
            else:
                token_type_ids = None
                
            if 'label' in data.keys():
                classify_labels = data['label'].to(device)
            else:
                classify_labels = None

            
            loss = model(
                        input_ids, 
                        attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, 
                        classify_labels=classify_labels, 
                        entity_idx=entity_idx,
                        use_cls=False
                    ) # mark
            
#             loss = model(
#                         input_ids, 
#                         attention_mask=attention_mask, 
#                         token_type_ids=token_type_ids, 
#                         classify_labels=classify_labels, 
#                         entity_idx=entity_idx,
#                         use_cls=True
#                     ) # cls

            total_train_loss += loss.item()
            loss.backward() 
            optimizer.step() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            scheduler.step() 
            model.zero_grad()
            
        avg_train_loss = total_train_loss / len(train_dataloader)  
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        print('Saveing Model...')
        torch.save(model, save_model)
        
        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=41)
    parser.add_argument("--data_path", type=str, default='data/semeval_data.csv')
    
    # setting
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased') 
    parser.add_argument("--seen_nums",type=int,default=5) 
    parser.add_argument("--save_result",type=str,default='output/')  

    
    # main
    args = parser.parse_args()
    if args.bert_model == 'roberta-base' or args.bert_model == 'roberta-large':
        args.max_length = 144
    max_length = args.max_length
    batch_size = args.batch_size
    epochs = args.epochs
    temp = args.temp # temperature hyper-parameter
    alpha = args.alpha
    dropout = args.dropout
    seeds = args.seeds
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    data_path = args.data_path
    model_name = args.bert_model
    gpu = args.gpu
    pretrained_model = args.bert_model
    
    labellist = get_labellist(args.data_path)
    random_order = list(range(len(labellist)))
    np.random.shuffle(random_order)
    train_labellist = [labellist[idx] for idx in random_order[:args.seen_nums]]
    test_labellist = [labellist[idx] for idx in random_order[args.seen_nums:]]
    unseen_nums = len(test_labellist) 
    seen_nums = len(train_labellist)
    save_model = os.path.join(args.save_result, 'model_unseen{}.pt'.format(unseen_nums))
    save_eval_result = os.path.join(args.save_result, 'eval_result.txt')

    # tokenizer initialization & add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model, never_split=['<e1>', '</e1>', '<e2>', '</e2>'])
    special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']}  # add special token
    tokenizer.add_special_tokens(special_tokens_dict)

    #data
    seen_sentences, seen_labels, seen_label_ids, label2id = read_data(data_path, train_labellist, use_mark=True)
    unseen_sentences, unseen_labels, unseen_label_ids, unseen_dict = read_data(data_path, test_labellist, use_mark=True)
    contrastive_sentence_pairs, contrastive_labels = get_contrastive_data(seen_sentences, seen_label_ids)
    #feature
    contrastive_features = get_contrastive_feature(contrastive_sentence_pairs, tokenizer, max_length)
    contrastive_entity_idx = get_entity_idx(contrastive_features, tokenizer)
    contrastive_dataset = RCLdataset(contrastive_features, contrastive_entity_idx, contrastive_labels)
    model = RCL(pretrained_model, temp, device, len(label2id), dropout=dropout, special_tokenizer=tokenizer)
    
    # train
    train(contrastive_dataset, model, device, train_batch_size=batch_size, train_epochs=epochs, seeds=seeds, save_model=save_model, collate_fn=collate_fn)
    
    # test
    t0 = time.time()
    model = torch.load(save_model)
    model.eval()  
    sent_embs = []
    with torch.no_grad():
        e1_tks_id = tokenizer.convert_tokens_to_ids('<e1>')
        e2_tks_id = tokenizer.convert_tokens_to_ids('<e2>')
        for unseen_sentence in unseen_sentences:
            sent_feature = tokenizer(
                    unseen_sentence,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors='pt'
                )
            input_ids = sent_feature['input_ids'].to(device)
            attention_mask = sent_feature['attention_mask'].to(device)
            if 'token_type_ids' in sent_feature.keys():
                token_type_ids = sent_feature['token_type_ids'].to(device)
            else:
                token_type_ids = None
            e1_idx = (sent_feature['input_ids'][0] == e1_tks_id).nonzero().flatten().tolist()[0]  
            e2_idx = (sent_feature['input_ids'][0] == e2_tks_id).nonzero().flatten().tolist()[0] 
            entity_idx = (e1_idx, e2_idx)

            sent_emb = model.encode(input_ids, attention_mask, token_type_ids=token_type_ids, entity_idx=entity_idx, use_cls=False)  # mark
    #         sent_emb = model.encode(input_ids, attention_mask, token_type_ids=token_type_ids, entity_idx=entity_idx, use_cls=True)  # cls
            sent_emb = sent_emb.detach().cpu().numpy()
            sent_embs.append(sent_emb[0])

    validation_time = format_time(time.time() - t0)
    print("  Validation took: {:}".format(validation_time))

    t0 = time.time()
    sent_embs = torch.tensor(sent_embs)
    print("data dimension is {}. ".format(sent_embs.shape[-1]))
    clusters = KMeans(n_clusters=unseen_nums, n_init=20)  #kmeans
    predict_labels = clusters.fit_predict(sent_embs)
    
    
    # evaluation
    metric_result = standard_metric_result(unseen_labels, predict_labels, labellist)
    # B3
    print('pretrained class eval')
    cluster_eval = ClusterEvaluation(unseen_label_ids, predict_labels).printEvaluation()
    print('B3', cluster_eval)
    # NMI, ARI, V_measure
    nmi = normalized_mutual_info_score
    print('NMI', nmi(unseen_label_ids, predict_labels))
    print('ARI', adjusted_rand_score(unseen_label_ids, predict_labels))
    print('Homogeneity', homogeneity_score(unseen_label_ids, predict_labels))
    print('Completeness', completeness_score(unseen_label_ids, predict_labels))
    print('V_measure', v_measure_score(unseen_label_ids, predict_labels))

    B3_F1 = cluster_eval['F1']
    B3_precision = cluster_eval['precision']
    B3_recall = cluster_eval['recall']
    NMI = normalized_mutual_info_score(unseen_label_ids, predict_labels)
    ARI = adjusted_rand_score(unseen_label_ids, predict_labels)
    Homogeneity = homogeneity_score(unseen_label_ids, predict_labels)
    Completeness = completeness_score(unseen_label_ids, predict_labels)
    V_measure = v_measure_score(unseen_label_ids, predict_labels)

    evaluation_dict = {
        'test_labellist': test_labellist, 
        'B3_F1': B3_F1,
        'B3_precision': B3_precision,
        'B3_recall': B3_recall,
        'NMI': NMI,
        'ARI': ARI, 
        'Homogeneity': Homogeneity,
        'Completeness': Completeness,
        'V_measure': V_measure
    }
    evaluation_dict = json.dumps(evaluation_dict, indent=4)
    with open(save_eval_result, 'w') as f:
        f.write('seen nums: {0}, unseen nums: {1}'.format(seen_nums, unseen_nums))
        f.write('\n')
        f.write(metric_result)
        f.write('\n')
        f.write('\n')
        f.write(evaluation_dict)
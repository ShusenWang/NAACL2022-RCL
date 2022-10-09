import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig



class RCL(nn.Module):  
    def __init__(self, bert_model, temp, device, num_label=4, dropout=0.5, alpha=0.15, special_tokenizer=None):
        super(RCL, self).__init__()
        self.config = AutoConfig.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        if special_tokenizer is not None:
            self.model.resize_token_embeddings(len(special_tokenizer)) 
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
#         self.dense_mark = nn.Linear(self.config.hidden_size*2, self.config.hidden_size*2)
        self.activation = nn.Tanh()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.device = device
        
        # classification
        self.alpha = alpha
        self.classifier = nn.Linear(self.config.hidden_size*2, num_label)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, entity_idx=None, classify_labels=None, use_cls=False):
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len) (32*2, 32)
        outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        )
        # class
        
        bert_emb = outputs[0]
        se_length = bert_emb.size(1)
        bert_emb = bert_emb.view((batch_size, num_sent, se_length, bert_emb.size(-1)))
        bert_hidden = bert_emb[:, 0] #[bs, length, hidden]
        
        relation_hidden = []
        for i in range(len(entity_idx)):
            head_idx = entity_idx[i][0][0]
            tail_idx = entity_idx[i][0][1]
            cls_token = bert_hidden[i][0].view(1, -1)
            head_entity = bert_hidden[i][head_idx].view(1, -1)
            tail_entity = bert_hidden[i][tail_idx].view(1, -1)
            relation = torch.cat([head_entity, tail_entity], dim=-1)
            relation_hidden.append(relation)
        relation_hidden = torch.cat(relation_hidden, dim=0)      
        
        relation_hidden = self.dropout(self.activation(relation_hidden))
        logit = self.classifier(relation_hidden)
        loss_ce = nn.CrossEntropyLoss()
        ce_loss = loss_ce(logit, classify_labels.view(-1))
    
        # cls
        if use_cls:
            last_hidden = outputs[0] # last_hidden
            cls_hidden = last_hidden[:, 0]
            pooler_output = cls_hidden.view((batch_size, num_sent, cls_hidden.size(-1))) # (bs, num_sent, hidden)
            pooler_output = self.dense(pooler_output)
            pooler_output = self.activation(pooler_output)
            z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
                   
        # marker
        elif not use_cls and entity_idx is not None:
            last_hidden = outputs[0] # last_hidden [bs, sent_length, hidden]
            last_hidden = self.dense(last_hidden)
            last_hidden = self.activation(last_hidden)
            sent_length = last_hidden.size(1)
            last_hidden = last_hidden.view((batch_size, num_sent, sent_length, last_hidden.size(-1)))
            sent1_hidden, sent2_hidden = last_hidden[:, 0], last_hidden[:, 1]  #[bs, sent_length, hidden]
            z1 = []
            z2 = []
            for i in range(len(entity_idx)):
                sent1_head_idx, sent1_tail_idx = entity_idx[i][0][0], entity_idx[i][0][1]
                sent2_head_idx, sent2_tail_idx = entity_idx[i][1][0], entity_idx[i][1][1]
                #sent1
                sent1_head_entity = sent1_hidden[i][sent1_head_idx]
                sent1_tail_entity = sent1_hidden[i][sent1_tail_idx]
                #sent2
                sent2_head_entity = sent2_hidden[i][sent2_head_idx]
                sent2_tail_entity = sent2_hidden[i][sent2_tail_idx]
                
                sent1_relation_expresentation = torch.cat([sent1_head_entity, sent1_tail_entity], dim=-1)
                sent2_relation_expresentation = torch.cat([sent2_head_entity, sent2_tail_entity], dim=-1)
                z1.append(sent1_relation_expresentation.unsqueeze(0))
                z2.append(sent2_relation_expresentation.unsqueeze(0))
            z1 = torch.cat(z1, dim=0)
            z2 = torch.cat(z2, dim=0)
            
        cos_sim = self.cos(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temp
        con_labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        cl_loss = loss_fct(cos_sim, con_labels) 
        
        return ce_loss + self.alpha*cl_loss

    
    def encode(self, input_ids, attention_mask, token_type_ids=None, entity_idx=None, use_cls=False):
        outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        )
        if use_cls:
            cls_token = outputs[0][:, 0]
            pooler_output = self.dense(cls_token)
            sent_emb = self.activation(pooler_output)
            return sent_emb
        elif not use_cls and entity_idx is not None:
            last_hidden = outputs[0] # last_hidden [1, sent_length, hidden]
            #test
            last_hidden = self.dense(last_hidden)
            last_hidden = self.activation(last_hidden)
            
            head_idx = entity_idx[0]
            tail_idx = entity_idx[1]
            head_entity = last_hidden[0][head_idx]
            tail_entity = last_hidden[0][tail_idx]
            sent_emb = torch.cat([head_entity, tail_entity], dim=-1)
            return sent_emb.unsqueeze(0)
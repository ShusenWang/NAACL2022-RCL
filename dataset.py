import torch
from torch.utils.data import Dataset


class RCLdataset(Dataset):

    def __init__(self, contrastive_features, contrastive_entity_idx, contrastive_labels=None):
        super(RCLdataset, self).__init__()
        self.contrastive_features = contrastive_features
        self.contrastive_entity_idx = contrastive_entity_idx
        if contrastive_labels:
            self.contrastive_labels = contrastive_labels
        else:
            self.contrastive_labels = None
        
    def __getitem__(self, idx):
        contrastive_features = self.contrastive_features
        contrastive_entity_idx = self.contrastive_entity_idx
        
        if self.contrastive_labels:
            contrastive_labels = self.contrastive_labels
            output = {
                "input_ids": contrastive_features[idx]["input_ids"],
                "attention_mask": contrastive_features[idx]["attention_mask"],
                "entity_idx": contrastive_entity_idx[idx],
                "label": contrastive_labels[idx]
            }
            if "token_type_ids" in contrastive_features[idx].keys():
                output["token_type_ids"] = contrastive_features[idx]["token_type_ids"]
        else:
            output = {
                "input_ids": contrastive_features[idx]["input_ids"],
                "attention_mask": contrastive_features[idx]["attention_mask"],
                "entity_idx": contrastive_entity_idx[idx]
            }
            if "token_type_ids" in contrastive_features[idx].keys():
                output["token_type_ids"] = contrastive_features[idx]["token_type_ids"]
                    
        return output
        

    def __len__(self): 
        return len(self.contrastive_features)


def collate_fn(batch):
    """
    对feature，label转为tensor
    """
    
    batch_input_ids = [data["input_ids"].unsqueeze(0) for data in batch]
    batch_attention_mask = [data["attention_mask"].unsqueeze(0) for data in batch]
    batch_entity_idx = [data["entity_idx"] for data in batch]
    if "token_type_ids" in batch[0].keys():
        batch_token_type_ids = [data["token_type_ids"].unsqueeze(0) for data in batch]
        batch_token_type_ids = torch.cat(batch_token_type_ids, dim=0)
    else:
        batch_token_type_ids = None
    
    
    batch_input_ids = torch.cat(batch_input_ids, dim=0)
    batch_attention_mask = torch.cat(batch_attention_mask, dim=0)
    batch_entity_idx = torch.tensor(batch_entity_idx)
    
    if "label" in batch[0].keys():
        batch_labels = [data["label"] for data in batch]
        batch_labels = torch.tensor(batch_labels)
        output = {
            'input_ids': batch_input_ids,  
            'attention_mask': batch_attention_mask,
            'entity_idx': batch_entity_idx,
            'label': batch_labels
           }
        if batch_token_type_ids is not None:
            output['token_type_ids'] = batch_token_type_ids
    else:
        output = {
            'input_ids': batch_input_ids, 
            'token_type_ids': batch_token_type_ids, 
            'attention_mask': batch_attention_mask,
            'entity_idx': batch_entity_idx
           }
        if batch_token_type_ids is not None:
            output['token_type_ids'] = batch_token_type_ids
    
    return output
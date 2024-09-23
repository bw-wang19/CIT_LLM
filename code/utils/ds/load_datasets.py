
# @file /workspace/acl/code/utils/ds/load_datasets.py

import transformers
import torch
import datasets
from torch.utils.data import (
    Dataset,
    random_split 
)
from torch.utils.data.dataloader import DataLoader


class MyDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        pass
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass
        

class Data_Loader():
    def __init__(self, *args, **kwargs):
        pass
    
    def Load_Datasets(self, *args, **kwargs):
        pass


def collate_func(batch: list[tuple]) -> None:
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding='max_length', 
                       truncation=True, return_tensors='pt')
    inputs['labels'] = torch.tensor(labels)
    return inputs
    
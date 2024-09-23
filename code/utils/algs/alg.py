import numpy
import os
from typing import List, Dict, Any
import transformers
import torch
from .LM_Cocktail import *
from .ft import *
import model

class Base_Alg():
    def __init__(self):
        pass
    
    
class Alg_LM_Cocktail(Base_Alg):
    def __init__(self, *args, **kwargs):
        super.__init__()
        self.base_model = None
        self.base_model_name = None
        pass
    
    def set_base_model(self, model_path, model_name):
        self.base_model = model.load_model(model_path)
        self.base_model_name = model_name
    
    def set_up(self, base_model, dataset_list:List[str], out_dir:str='./cache_LM_Cocktail', ft_method=ft_default, *args, **kwargs):
        
        
        new_models_list = []
        for idx, dataset_name in enumerate(dataset_list):
            data_loader = build_loader(dataset_name)
            new_model = ft_method(self.base_model, data_loader, ft_args)
            save_path = os.path.join(out_dir, self.base_model_name+dataset_name)
            new_model.save_pretrained(save_path)
            new_models_list.append(save_path)
        return new_models_list
    
    def train(self, model, train_loader, val_loader, merge_type, *args, **kwargs):
        pass
    
    
    

def init(alg_name:str, ) -> Base_Alg:
    if alg_name == 'LM_Cocktail':
        alg = Alg_LM_Cocktail()
    else:
        raise ValueError('Unsupported algorithm')
    
    return alg


     
    
    
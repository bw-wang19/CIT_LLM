import numpy
import os
from typing import List, Dict, Any
import transformers
import torch
import LM_Cocktail
import LM_Cocktail.utils
import utils.peft.finetune as ft
import model
import copy
from random import *

class Base_Alg():
    def __init__(self):
        pass
    
    
class Alg_LM_Cocktail(Base_Alg):
    def __init__(self, *args, **kwargs):
        super.__init__()
        self.base_model = None
        self.base_model_name = None
        self.base_model_type = None
        pass
    
    def set_base_model(self, model_path:str, model_name:str, model_type:str='decoder'):
        self.base_model = model.load_model(model_path)
        self.base_model_name = model_name
        self.base_model_type = model_type
        self.base_model_path = model_path
    
    def set_up(self, 
               dataset_list:List[str],
               sys_config,
               ft_config, 
               out_dir:str='./cache_LM_Cocktail', 
               ft_method=ft.ft_llama_factory, *args, **kwargs):
        new_models_list = []
        for idx, dataset_name in enumerate(dataset_list):
            data_loader = build_loader(dataset_name)
            ### TODO
            new_model = ft_method(model_path=self.base_model_path, 
                                  sys_config=sys_config, 
                                  ft_config=ft_config, 
                                  *args, **kwargs)
            save_path = os.path.join(out_dir, self.base_model_name+dataset_name)
            new_model.save_pretrained(save_path)
            new_models_list.append(save_path)
            return new_models_list
    
    def train(self, train_loader, val_loader, example_data:List[Dict], 
              ft_args, models_d_path:List[str],
              ft_method=ft.ft_llama_factory, alpha=0.5, 
              out_dir='./new_model', *args, **kwargs):
        # fine-tune on dataset
        model_t = ft_method(self.base_model, train_loader, val_loader, ft_args)
        
        # get merge model on domain models
        model_m = LM_Cocktail.mix_models_with_data(model_names_or_paths = models_d_path,
                             model_type = self.base_model_type,
                             example_data = example_data,
                             temperature = kwargs['temperature'],
                             batch_size = kwargs['batch_size'],
                             max_input_length = kwargs['max_input_length'],
                             neg_number = kwargs['neg_number'],
                             output_path = None)
        
        model_t_param = model_t.state_dict()
        model_m_param = model_m.state_dict()
        model_param_list = [model_t_param, model_m_param]
        model_r_param = LM_Cocktail.utils.merge_param(model_param_list = model_param_list,
                                                      weights = [alpha, 1-alpha])
        model_r = copy.deepcopy(self.base_model)
        model_r.load_state_dict(model_r_param)
        model_r.save_pretrained(out_dir)
        return model_r
    
    def generate_example(dataset, example_num:int=5, input_key='input', output_key='output') -> List[Dict]:
        example_data = []
        data_num = dataset.length()
        sample_idx_list = sample([i for i in range(data_num)], example_num)
        for idx in sample_idx_list:
            data = dataset[idx]
            example_data.append(
                {
                    "input" : data[input_key],
                    "output" : data[output_key]
                }
            )
        return example_data
        
    

def init(alg_name:str, ) -> Base_Alg:
    if alg_name == 'LM_Cocktail':
        alg = Alg_LM_Cocktail()
    else:
        raise NotImplementedError('Unsupported algorithm')
    
    return alg


     
    
    
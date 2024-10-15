import transformers
import os
import subprocess
import yaml
from typing import Dict, List



def update_yaml(yaml_path:str, config:Dict) -> None:
    
    with open(file=yaml_path, mode="w") as f:
        cfg_yaml = yaml.safe_load(f)
        for key in cfg_yaml.keys():
            if key in config:
                cfg_yaml[key] = config[key]
        yaml.dump(cfg_yaml, f)



def ft_llama_factory(model_path:str, sys_config, ft_config, 
                     ft_yaml_path='/workspace/acl/configs/yaml/llama2_7B_lora_sft.yaml', 
                     merge_yaml_path='/workspace/acl/configs/yaml/llam2_7B_lora_sft_merge.yaml', 
                     *args, **kwargs) -> None:
    '''
    do supervised fine-tuning with LlaMA-Factory API
    github:https://github.com/hiyouga/LLaMA-Factory
    tutorial:https://blog.csdn.net/python12345678_/article/details/140346926
    
    check bash params using 'llamafactory-cli train -h'
    
    
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \  
    --stage sft \  
    --do_train \  
    --model_name_or_path /media/codingma/LLM/llama3/Meta-Llama-3-8B-Instruct \  
    --dataset alpaca_gpt4_zh,identity,adgen_local \  
    --dataset_dir ./data \  
    --template llama3 \  
    --finetuning_type lora \  
    --lora_target q_proj,v_proj \  
    --output_dir ./saves/LLaMA3-8B/lora/sft \  
    --overwrite_cache \  
    --overwrite_output_dir \  
    --cutoff_len 1024 \  
    --preprocessing_num_workers 16 \  
    --per_device_train_batch_size 2 \  
    --per_device_eval_batch_size 1 \  
    --gradient_accumulation_steps 8 \  
    --lr_scheduler_type cosine \  
    --logging_steps 50 \  
    --warmup_steps 20 \  
    --save_steps 100 \  
    --eval_steps 50 \  
    --evaluation_strategy steps \  
    --load_best_model_at_end \  
    --learning_rate 5e-5 \  
    --num_train_epochs 5.0 \  
    --max_samples 1000 \  
    --val_size 0.1 \  
    --plot_loss \  
    --fp16
--------------
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \  
    --model_name_or_path /media/codingma/LLM/llama3/Meta-Llama-3-8B-Instruct \  
    --adapter_name_or_path ./saves/LLaMA3-8B/lora/sft  \  
    --template llama3 \  
    --finetuning_type lora \  
    --export_dir megred-model-path \  
    --export_size 2 \  
    --export_device cpu \  
    --export_legacy_format False
    '''
    
    # update yaml config file
    update_yaml(ft_yaml_path, ft_config)
    update_yaml(merge_yaml_path, ft_config)
    
    # fine tuning
    cuda_devices_cmd = 'CUDA_VISIBLE_DEVICES=' + sys_config['devices']
    bash_cmd_ft = cuda_devices_cmd + ' llamafactory-cli train ' + ft_yaml_path
    os.system(bash_cmd_ft)
    
    # merge original model with adapters
    bash_cmd_merge = cuda_devices_cmd + ' llamafactory-cli export ' + merge_yaml_path
    os.system(bash_cmd_merge)
    
    

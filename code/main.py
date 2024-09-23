# @file /workspace/acl/code/main.py

import logging
import os
import sys
import json
import time

import torch
import datasets
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datasets import load_dataset


from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig

from utils import (
    ds,
    Trainer, 
    Test,
)





from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )


@dataclass
class SystemArguments:
    '''
    Arguments for System setting
    '''
    seed: int = field(
        default=0,
        metadata={"help": "set random seed"}
    )
    log_path: Optional[str] = field(
        default = None,
        metadata={"help": "log file path"}
    )
    devices: str = field(
        default = '0,1,2,3,4,5,6,7',
        metadata = {
            "help": "available gpu-id, \'\'for using cpu"
        }
    )

@dataclass
class ModelArguments:
    '''
    Arguments for pretrained model to load
    '''
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model"}
    )
    model_type: str = field(
        metadata={"help": "Type of pretrained model"}
    )
    config_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Config path"}
    )
    tokenizer_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Tokenizer path"}
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "rank of lora matrix"}
    )

@dataclass
class DataArguments:
    '''
    Arguments for settings of datasets
    '''
    phase_num: Optional[int] = field(
        default=None,
        metadata={"help": "phase number for continual learning"}
    )
    sequence_file: str = field(
        metadata={"help": "file path of sequence of datasets"}
    ),
    max_train_num_per_task: int = field(
        default=10000, 
        metadata={"help": "The maximum number of accessable instances for each task training."}
    )
    
    
    pass
    
@dataclass
class FTArguments:
    '''
    Arguments for instruction tuning stage
    '''
    max_train_num_per_task: int = field(
        default=10000, 
        metadata={"help": "The maximum number of accessable instances for each task training."}
    )
    max_eval_num_per_task: int = field(
        default=200,
        metadata={"help": "The maximum number of accessable instances for each task evaluation."}
    )
    train_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "data ratio for training"}
    )
    eval_ration: Optional[float] = field(
        default=None,
        metadata={"help": "data ratio for evalution"}
    )
    
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    pass

@dataclass
class TrainArguments:
    '''
    Arguments for Training stage
    '''
    batch_size: int = field(
        default = 32,
        metadata = {
            "help": "batch size"
        }
    )
    max_epoch: int = field(
        default = 3,
        metadata = {
            "help": "max epochs of training stage"
        }
    )
    optimizer: str = field(
        default = "SGD",
        metadata = {
            "help": "optimizer for gradient descent"
        }
    )
    lr: float = field(
        default = 1e-4,
        metadata = {
            "help": "learning rate for gradient descent"
        }
    )
    beta1: Optional[float] = field(
        default = 0.1,
        metadata = {
            "help": "beta1 for SGDM"
        },
    )
    beta2: Optional[float] = field(
        default = 0.01,
        metadata = {
            "help": "beta2 for Adam"
        } 
    )
    pass

@dataclass
class EvalArguments:
    pass

def main():
    # parse arguments
    parser = HfArgumentParser((SystemArguments, 
                               ModelArguments, 
                               DataArguments, 
                               FTArguments, 
                               TrainArguments, 
                               EvalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        sys_args, model_args, data_args, ft_args, train_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        sys_args, model_args, data_args, ft_args, train_args, eval_args = parser.parse_args_into_dataclasses()
    # system set
    set_seed(sys_args.seed)
    
    # load model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_r,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q", "v"],
        bias="none"
    )
    
    if model_args.model_type == 'CausalLM':
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    else:
        pass
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    
    
    # load dataset
    dataset_loader = ds.Data_Loader(data_args)
    
    # load CL_algs
    alg = algs.Algs_Init(train_args)
    
    # Train
    trainer = Trainer(model, dataset_loader, alg)
    trainer.train(train_args)
    
    
    # Test on benchmarks
    
    Test(model, eval_args)
    
    pass





if __name__ == "__main__":
    main()
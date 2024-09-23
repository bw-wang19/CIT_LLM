import numpy as np
import transformers
from transformers import(LlamaForCausalLM, 
                         LlamaTokenizer,
                         AutoTokenizer, 
                         AutoModelForCausalLM,
                         AutoModel,
                         AutoModelForSequenceClassification,
                         AutoModelForSeq2SeqLM,
                         is_torch_npu_available
)



def load_model(model_name_or_path:str, model_type:str, device_map="auto", trust_remote_code:bool=True):
    if model_type == 'decoder':
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, device_map = device_map)
    elif model_type == 'encoder':
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, device_map = device_map)
    elif model_type == 'reranker':
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, device_map = device_map)
    elif model_type == 'encoder-decoder':      
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    else:
        raise NotImplementedError(f"not support this model_type: {model_type}")
    return model

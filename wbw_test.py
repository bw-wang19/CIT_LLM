import transformers
import torch
import numpy as np

from transformers import (LlamaForCausalLM, 
                          LlamaTokenizer,
                          AutoModel,
                          AutoTokenizer)

model_path = '/workspace/acl/model_zoo/llama/llama-2-7b-hf'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
inputs = tokenizer('Hello, my dog is cute', return_tensors='pt')
# 在模型上进行推理
outputs = model(**inputs)
print(tokenizer.decode(outputs))

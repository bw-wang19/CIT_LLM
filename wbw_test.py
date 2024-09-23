import transformers
import torch
import numpy as np

from transformers import LlamaForCausalLM, LlamaTokenizer

model_path = '/workspace/acl/model_zoo/llama/llama-2-7b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)
inputs = tokenizer('Hello, my dog is cute', return_tensors='pt')
# 在模型上进行推理
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

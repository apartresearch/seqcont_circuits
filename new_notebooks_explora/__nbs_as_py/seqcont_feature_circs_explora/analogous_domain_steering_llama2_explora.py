#!/usr/bin/env python
# coding: utf-8

# # setup

# ## load repo, libs, and functions

# In[1]:


get_ipython().run_cell_magic('capture', '', '# ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`\n# must do this at start of nb\n!pip install accelerate\n')


# In[2]:


# get the data and functions
get_ipython().system('git clone https://github.com/nrimsky/CAA.git')


# In[3]:


import sys
sys.path.append('/content/CAA')


# In[4]:


# see requirements.txt
get_ipython().system('pip install python-dotenv==1.0.0')


# In[5]:


# import generate_vectors
from generate_vectors import *


# In[6]:


import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS
)

load_dotenv()

# HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


# In[7]:


import torch.nn.functional as F


# # hf key

# Do this to avoid gated error (Llama 2 needs permission from meta to grant your token key access)

# In[8]:


from huggingface_hub import notebook_login
notebook_login()  # place token here


# In[9]:


# os.environ['HUGGINGFACE_CO_API_KEY'] = '' # token here


# # Load Model

# ## gpt2

# In[ ]:


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


# model_name = "gpt2-medium"
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# ## Llama-7b

# In[10]:


use_base_model = True
model_size = '7b'
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

model = LlamaWrapper(
        HUGGINGFACE_TOKEN, size=model_size, use_chat=not use_base_model
    )
# model.set_save_internal_decodings(False)
# model.reset_all()


# If get: OutOfMemoryError: CUDA out of memory.
# 
# Try: low_cpu_mem_usage. Load without wrapper first, then instead of using wrapper's methods, paste the methods here to analyze their internals

# In[ ]:


# from transformers import LlamaForCausalLM, LlamaTokenizer
# LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-chat-hf"

# tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH)
# hf_model = LlamaForCausalLM.from_pretrained(LLAMA_2_7B_CHAT_PATH, low_cpu_mem_usage=True)


# # Generate by 2 prompts

# In[51]:


# do this b/c anger is one token, calm is 2, so this pads anger with 50256
# tokens = model.to_tokens(batch_input) # for Transformerlens

p_prompt = ["first"]
n_prompt = ["one"]

# , return_tensors="pt") is returns type. "pt" stands for PyTorch tensors
# model only takes pytorch tensors in, not list
p_tokens = model.tokenizer(p_prompt, return_tensors="pt")['input_ids']
p_tokens = p_tokens.to(model.device)
n_tokens = model.tokenizer(n_prompt, return_tensors="pt")['input_ids']
n_tokens = n_tokens.to(model.device)

print(p_tokens)
print(n_tokens)


# In[52]:


layers = [15]

model.reset_all()
model.get_logits(p_tokens)
for layer in layers:
    p_activations = model.get_last_activations(layer)
    p_activations = p_activations[0, -1, :].detach().cpu()
model.reset_all()
model.get_logits(n_tokens)
for layer in layers:
    n_activations = model.get_last_activations(layer)
    n_activations = n_activations[0, -1, :].detach().cpu()


# In[53]:


for layer in layers:
    vec = (p_activations - n_activations)


# In[54]:


vec.shape


# # Steer

# In[72]:


layer = 15
multiplier = 0
model.set_save_internal_decodings(True)
model.reset_all()
model.set_add_activations(layer, multiplier * vec.cuda())
prompt = ["second"]
tokens = model.tokenizer(prompt, return_tensors="pt")['input_ids']
tokens = tokens.to(model.device)

output = model.model(tokens)
print(len(output))
print(output[0].shape)
print(len(output[1]))


# In[74]:


output[0][0, -1].shape


# In[79]:


next_token = output[0][0, -1].argmax(dim=-1)
model.tokenizer.decode(next_token)
# next_char = model.model.to_string(next_token)
# next_char


# ## ranks to numwords

# In[83]:


layer = 15
multiplier = 0
model.set_save_internal_decodings(True)
model.reset_all()
model.set_add_activations(layer, multiplier * vec.cuda())
prompt = ["two"]
tokens = model.tokenizer(prompt, return_tensors="pt")['input_ids']
tokens = tokens.to(model.device)

output = model.model(tokens)
next_token = output[0][0, -1].argmax(dim=-1)
model.tokenizer.decode(next_token)


# # loop thru layers to find steer vecs

# # first to one

# In[130]:


p_prompt = ["first"]
n_prompt = ["one"]

# , return_tensors="pt") is returns type. "pt" stands for PyTorch tensors
# model only takes pytorch tensors in, not list
p_tokens = model.tokenizer(p_prompt, return_tensors="pt")['input_ids']
p_tokens = p_tokens.to(model.device)
n_tokens = model.tokenizer(n_prompt, return_tensors="pt")['input_ids']
n_tokens = n_tokens.to(model.device)

print(p_tokens)
print(n_tokens)


# In[131]:


layers = [i for i in range(0, 31)]
pos_activations = dict([(layer, []) for layer in layers])
neg_activations = dict([(layer, []) for layer in layers])
vec_layer_dict = dict([(layer, []) for layer in layers])

model.reset_all()
model.get_logits(p_tokens)
for layer in layers:
    p_activations = model.get_last_activations(layer)
    p_activations = p_activations[0, -1, :].detach().cpu()
    pos_activations[layer] = p_activations
model.reset_all()
model.get_logits(n_tokens)
for layer in layers:
    n_activations = model.get_last_activations(layer)
    n_activations = n_activations[0, -1, :].detach().cpu()
    neg_activations[layer] = n_activations

for layer in layers:
    p_activations = pos_activations[layer]
    n_activations = neg_activations[layer]
    vec = (p_activations - n_activations)
    vec_layer_dict[layer] = vec


# In[132]:


# prompt = ["first second third "]
prompt = ["first"]
tokens = model.tokenizer(prompt, return_tensors="pt")['input_ids']
tokens = tokens.to(model.device)

for layer in layers:
    model.set_save_internal_decodings(True)
    model.reset_all()

    multiplier = 5
    vec = vec_layer_dict[layer]
    model.set_add_activations(layer, multiplier * vec.cuda())

    output = model.model(tokens)
    next_token = output[0][0, -1].argmax(dim=-1)
    next_char = model.tokenizer.decode(next_token)
    print(layer, next_char)


# In[135]:


prompt = ["first second third "]
tokens = model.tokenizer(prompt, return_tensors="pt")['input_ids']
tokens = tokens.to(model.device)

for layer in layers:
    print(layer)

    # model.set_save_internal_decodings(True)
    model.reset_all()
    print("Before steering:")

    output_tokens = model.model.generate(tokens, max_length= 10)
    generated_text = model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(generated_text)

    model.reset_all()
    print("After steering:")

    multiplier = 5
    vec = vec_layer_dict[layer]
    model.set_add_activations(layer, multiplier * vec.cuda())

    output_tokens = model.model.generate(tokens, max_length= 10)
    generated_text = model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(generated_text)


# # love to hate

# In[136]:


p_prompt = ["love"]
n_prompt = ["hate"]

# , return_tensors="pt") is returns type. "pt" stands for PyTorch tensors
# model only takes pytorch tensors in, not list
p_tokens = model.tokenizer(p_prompt, return_tensors="pt")['input_ids']
p_tokens = p_tokens.to(model.device)
n_tokens = model.tokenizer(n_prompt, return_tensors="pt")['input_ids']
n_tokens = n_tokens.to(model.device)

print(p_tokens)
print(n_tokens)


# In[137]:


layers = [i for i in range(0, 31)]
pos_activations = dict([(layer, []) for layer in layers])
neg_activations = dict([(layer, []) for layer in layers])
vec_layer_dict = dict([(layer, []) for layer in layers])

model.reset_all()
model.get_logits(p_tokens)
for layer in layers:
    p_activations = model.get_last_activations(layer)
    p_activations = p_activations[0, -1, :].detach().cpu()
    pos_activations[layer] = p_activations
model.reset_all()
model.get_logits(n_tokens)
for layer in layers:
    n_activations = model.get_last_activations(layer)
    n_activations = n_activations[0, -1, :].detach().cpu()
    neg_activations[layer] = n_activations

for layer in layers:
    p_activations = pos_activations[layer]
    n_activations = neg_activations[layer]
    vec = (p_activations - n_activations)
    vec_layer_dict[layer] = vec


# In[138]:


prompt = ["I hate you because"]
# prompt = ["love"]
# prompt = ["hate"]
tokens = model.tokenizer(prompt, return_tensors="pt")['input_ids']
tokens = tokens.to(model.device)

for layer in layers:
    # model.set_save_internal_decodings(True)
    model.reset_all()

    multiplier = 5
    vec = vec_layer_dict[layer]
    model.set_add_activations(layer, multiplier * vec.cuda())

    output = model.model(tokens)
    next_token = output[0][0, -1].argmax(dim=-1)
    next_char = model.tokenizer.decode(next_token)
    print(layer, next_char)


# In[139]:


model.reset_all()
output_tokens = model.model.generate(tokens, max_length=10)
generated_text = model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
generated_text


# In[140]:


prompt = ["I hate you because "]
tokens = model.tokenizer(prompt, return_tensors="pt")['input_ids']
tokens = tokens.to(model.device)

for layer in layers:
    print(layer)

    # model.set_save_internal_decodings(True)
    model.reset_all()
    print("Before steering:")

    output_tokens = model.model.generate(tokens, max_length= 10)
    generated_text = model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(generated_text)

    model.reset_all()
    print("After steering:")

    multiplier = 5
    vec = vec_layer_dict[layer]
    model.set_add_activations(layer, multiplier * vec.cuda())

    output_tokens = model.model.generate(tokens, max_length= 10)
    generated_text = model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(generated_text)


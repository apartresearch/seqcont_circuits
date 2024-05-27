#!/usr/bin/env python
# coding: utf-8

# # Change Inputs Here

# In[1]:


task = "numerals"  # choose: numerals, numwords, months
prompt_types = ['done', 'lost', 'names']
num_samps_per_ptype = 512 #768 512

model_name = "gpt2-small"

save_files = True
run_on_other_tasks = True


# # Setup

# In[2]:


get_ipython().run_cell_magic('capture', '', '%pip install git+https://github.com/neelnanda-io/TransformerLens.git\n')


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
# import plotly.express as px
from torch.utils.data import DataLoader

from jaxtyping import Float, Int
from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import pickle
from google.colab import files

import matplotlib.pyplot as plt
import statistics


# In[4]:


import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache


# We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.

# In[5]:


torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## Load Model

# In[6]:


model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)


# ## Import functions from repo

# In[13]:


get_ipython().system('git clone https://github.com/apartresearch/seqcont_circuits.git')
get_ipython().run_line_magic('cd', '/content/seqcont_circuits/src/iter_node_pruning')


# In[14]:


## comment this out when debugging functions in colab to use funcs defined in colab

from dataset import Dataset
from metrics import *
from head_ablation_fns import *
from mlp_ablation_fns import *
from node_ablation_fns import *
from loop_node_ablation_fns import *


# # Load datasets

# In[16]:


class Dataset:
    def __init__(self, prompts, pos_dict, tokenizer):  # , S1_is_first=False
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.N = len(prompts)
        self.max_len = max(
            [
                len(self.tokenizer(prompt["text"]).input_ids)
                for prompt in self.prompts
            ]
        )
        all_ids = [0 for prompt in self.prompts] # only 1 template
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])

        texts = [ prompt["text"] for prompt in self.prompts ]
        self.toks = torch.Tensor(self.tokenizer(texts, padding=True).input_ids).type(
            torch.int
        )
        self.corr_tokenIDs = [
            # self.tokenizer.encode(" " + prompt["corr"])[0] for prompt in self.prompts
            self.tokenizer.encode(prompt["corr"])[0] for prompt in self.prompts
        ]
        self.incorr_tokenIDs = [
            # self.tokenizer.encode(" " + prompt["incorr"])[0] for prompt in self.prompts
            self.tokenizer.encode(prompt["incorr"])[0] for prompt in self.prompts
        ]

        # word_idx: for every prompt, find the token index of each target token and "end"
        # word_idx is a tensor with an element for each prompt. The element is the targ token's ind at that prompt
        self.word_idx = {}
        # for targ in [key for key in self.prompts[0].keys() if (key != 'text' and key != 'corr' and key != 'incorr')]:
        for targ in [key for key in pos_dict]:
            targ_lst = []
            for prompt in self.prompts:
                input_text = prompt["text"]
                tokens = self.tokenizer.tokenize(input_text)
                # if S1_is_first and targ == "S1":  # only use this if first token doesn't have space Ġ in front
                #     target_token = prompt[targ]
                # else:
                #     target_token = "Ġ" + prompt[targ]
                # target_index = tokens.index(target_token)
                target_index = pos_dict[targ]
                targ_lst.append(target_index)
            self.word_idx[targ] = torch.tensor(targ_lst)

        targ_lst = []
        for prompt in self.prompts:
            input_text = prompt["text"]
            tokens = self.tokenizer.tokenize(input_text)
            end_token_index = len(tokens) - 1
            targ_lst.append(end_token_index)
        self.word_idx["end"] = torch.tensor(targ_lst)

    def __len__(self):
        return self.N


# In[17]:


# prompts_list = []

# for i in prompt_types:
#     file_name = f'/content/seqcont_circuits/data/{task}/{task}_prompts_{i}.pkl'
#     with open(file_name, 'rb') as file:
#         filelist = pickle.load(file)

#     print(filelist[0]['text'])
#     prompts_list += filelist [:num_samps_per_ptype]

# len(prompts_list)


# In[18]:


def generate_prompts_list(x ,y):
    prompts_list = []
    for i in range(x, y):
        prompt_dict = {
            'S1': str(i),
            'S2': str(i+1),
            'S3': str(i+2),
            'S4': str(i+3),
            'corr': str(i+4),
            'incorr': str(i+3),
            'text': f"{i} {i+1} {i+2} {i+3}"
        }
        prompts_list.append(prompt_dict)
    return prompts_list

prompts_list = generate_prompts_list(1, 2)
prompts_list


# In[19]:


pos_dict = {}
for i in range(len(model.tokenizer.tokenize(prompts_list[0]['text']))):
    pos_dict['S'+str(i)] = i


# In[20]:


dataset = Dataset(prompts_list, pos_dict, model.tokenizer)


# In[21]:


# file_name = f'/content/seqcont_circuits/data/{task}/randDS_{task}.pkl'
# with open(file_name, 'rb') as file:
#     prompts_list_2 = pickle.load(file)


# In[22]:


import random

def generate_prompts_list_corr(prompt_list):
    outlist = []
    # for i in range(100):
    for prompt_dict in prompts_list:
        r1 = random.randint(1, 12)
        r2 = random.randint(1, 12)
        while True:
            r3 = random.randint(1, 12)
            r4 = random.randint(1, 12)
            if r4 - 1 != r3:
                break
        new_text = prompt_dict['text'].replace(prompt_dict['S1'], str(r1)).replace(prompt_dict['S2'], str(r2)).replace(prompt_dict['S3'], str(r3)).replace(prompt_dict['S4'], str(r4))
        new_prompt_dict = {
            'S1': str(r1),
            'S2': str(r2),
            'S3': str(r3),
            'S4': str(r4),
            'corr': prompt_dict['corr'],
            'incorr': prompt_dict['incorr'],
            'text': new_text
        }
        outlist.append(new_prompt_dict)
    return outlist
prompts_list_2 = generate_prompts_list_corr(prompts_list)
len(prompts_list_2)


# In[23]:


dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)


# ## Get orig score

# In[24]:


model.reset_hooks(including_permanent=True)
logits_original = model(dataset.toks)
orig_score = get_logit_diff(logits_original, dataset)
orig_score


# In[25]:


import gc

del(logits_original)
torch.cuda.empty_cache()
gc.collect()


# # Generate- Unablated

# Generate output in GPT-2 ()

# In[7]:


reference_text = "What comes after Monday is Tuesday, and two days after is"
tokens = model.to_tokens(reference_text).to(device)

logits, cache = model.run_with_cache(tokens)
# probs = logits.softmax(dim=-1)


# In[8]:


next_token = logits[0, -1].argmax(dim=-1)  # logits have shape [1, sequence_length, vocab_size]
next_char = model.to_string(next_token)
print(repr(next_char))


# In[10]:


model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

print(f"Sequence so far: {model.to_string(tokens)[0]!r}")

corr_ans_tokLen = 0
for i in range(10):
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = torch.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = model(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = model.to_string(next_token)
    corr_ans_tokLen += 1
    if next_char == ' ':
        break


# In[11]:


# first, we get how long the correct string answer is in terms of token IDs

corr_ans_tokLen


# # Generate- Ablated

# then, for model after hook, we keep on passing in the output autoregressively for the number of tokens the correct answer is. each pass, we measure the logit of the correct token of that pass, taking the difference with the logit of the last sequence member

# In[26]:


## heads_not_ablate is components to keep
# heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
# heads_not_ablate = [(9, 1)]
heads_not_ablate = []  # ablate all heads but not MLPs
mlps_not_ablate = []  # ablate all MLPs

# CIRCUIT = {}
# SEQ_POS_TO_KEEP = {}
# for i in range(len(model.tokenizer.tokenize(dataset_2.prompts[0]['text']))):
#     CIRCUIT['S'+str(i)] = lst
#     if i == len(model.tokenizer.tokenize(dataset_2.prompts[0]['text'])) - 1:
#         SEQ_POS_TO_KEEP['S'+str(i)] = 'end'
#     else:
#         SEQ_POS_TO_KEEP['S'+str(i)] = 'S'+str(i)

model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)
# model = add_ablation_hook_head(model, means_dataset=dataset_2, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)
logits_minimal = model(dataset.toks)

new_score = get_logit_diff(logits_minimal, dataset)
new_score


# In[27]:


print(f"Sequence so far: {model.to_string(tokens)[0]!r}")

corr_ans_tokLen = 0
for i in range(10):
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = torch.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = model(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = model.to_string(next_token)
    corr_ans_tokLen += 1
    if next_char == ' ':
        break


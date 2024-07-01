#!/usr/bin/env python
# coding: utf-8

# # Setup

# ## Change Inputs Here

# In[1]:


model_name = "gpt2-small"
save_files = True


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


import pdb


# ## Load Model

# In[6]:


torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[7]:


model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)


# ## Import functions from repo

# In[8]:


get_ipython().system('git clone https://github.com/apartresearch/seqcont_circuits.git')
get_ipython().run_line_magic('cd', '/content/seqcont_circuits/src/iter_node_pruning')


# In[9]:


## comment this out when debugging functions in colab to use funcs defined in colab

# don't improt this
# # from dataset import Dataset

from metrics import *
from head_ablation_fns import *
from mlp_ablation_fns import *
from node_ablation_fns import *
from loop_node_ablation_fns import *


# ## fns

# In[10]:


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


# In[11]:


def generate_prompts_list_longer(text, tokens):
    prompts_list = []
    prompt_dict = {
        'corr': str(1),
        'incorr': str(2),
        'text': text}
    tokens_as_strs = model.tokenizer.tokenize(text)
    # for i in range(tokens.shape[1]):
    for i, tok in enumerate(tokens_as_strs):
        prompt_dict['S'+str(i)] = tok
    prompts_list.append(prompt_dict)
    return prompts_list


# # Load datasets

# In[12]:


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


# In[13]:


pos_dict = {}
for i in range(len(model.tokenizer.tokenize(prompts_list[0]['text']))):
    pos_dict['S'+str(i)] = i


# In[14]:


dataset = Dataset(prompts_list, pos_dict, model.tokenizer)


# In[15]:


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


# In[16]:


dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)


# ## Get orig score

# In[17]:


model.reset_hooks(including_permanent=True)
logits_original = model(dataset.toks)
orig_score = get_logit_diff(logits_original, dataset)
orig_score


# In[18]:


import gc

del(logits_original)
torch.cuda.empty_cache()
gc.collect()


# # logit diff for mult tok answers

# In[19]:


def clean_gen(model, clean_text, corr_ans):
    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
    tokens = model.to_tokens(clean_text).to(device)
    tokens = tokens[:, 1:] # get rid of prepend bos when using model.to_tokens

    total_score = 0
    corr_ans_tokLen = 0
    ans_so_far = ''
    # while True:
    for i in range(5):
        print(f"Sequence so far: {model.to_string(tokens)[0]!r}")
        logits = model(tokens)
        next_token = logits[0, -1].argmax(dim=-1) # Get the predicted token at the end of our sequence
        next_char = model.to_string(next_token)

        dataset = Dataset(prompts_list, pos_dict, model.tokenizer)

        # new_score = get_logit_diff(logits, dataset)

        # measure how far away predicted logit is from corr token?

        # corr_logits = logits[:, dataset.word_idx["end"], dataset.corr_tokenIDs]
        # incorr_logits = logits[:, dataset.word_idx["end"], dataset.incorr_tokenIDs]
        # new_score = corr_logits - incorr_logits

        corr_logits = logits[:, -1, next_token]
        total_score += corr_logits
        print(f"logit diff of new char: {corr_logits}")

        ans_so_far += next_char
        corr_ans_tokLen += 1
        print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
        if ans_so_far == corr_ans:
            print('\nTotal logit diff: ', total_score.item())
            break
        # Define new input sequence, by appending the previously generated token
        tokens = torch.cat([tokens, next_token[None, None]], dim=-1)
    return corr_ans_tokLen


# In[20]:


clean_text = "1 2 3 4"
corr_ans = ' 5'
corr_ans_tokLen = clean_gen(model, clean_text, corr_ans)


# In[21]:


def ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen):
    tokens = model.to_tokens(clean_text).to(device)
    prompts_list = generate_prompts_list_longer(clean_text, tokens)

    corr_tokens = model.to_tokens(corr_text).to(device)
    prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)

    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
    pos_dict = {}
    for i in range(len(model.tokenizer.tokenize(prompts_list_2[0]['text']))):
        pos_dict['S'+str(i)] = i
    dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)
    model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

    tokens = tokens[:, 1:] # get rid of prepend bos when using model.to_tokens
    logits = model(tokens)
    next_token = logits[0, -1].argmax(dim=-1)
    next_char = model.to_string(next_token)

    total_score = 0

    print(f"Sequence so far: {model.to_string(tokens)[0]!r}")
    for i in range(corr_ans_tokLen):
    # for i in range(5):
        print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

        clean_text = clean_text + next_char
        tokens = model.to_tokens(clean_text).to(device)
        tokens = tokens[:, 1:]
        print(clean_text)

        # get new ablation dataset
        model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

        corr_text = corr_text + next_char
        corr_tokens = model.to_tokens(corr_text).to(device)
        prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)
        print(corr_text)

        pos_dict = {}
        for i in range(len(model.tokenizer.tokenize(prompts_list_2[0]['text']))):
            pos_dict['S'+str(i)] = i

        dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)

        model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

        logits = model(tokens)
        next_token = logits[0, -1].argmax(dim=-1) # Get the predicted token at the end of our sequence
        next_char = model.to_string(next_token)

        print('\n')
        print(f"Sequence so far: {model.to_string(tokens)[0]!r}")

        new_score = get_logit_diff(logits, dataset)
        total_score += new_score
        print(f"corr logit of new char: {new_score}")
    print('\n Total corr logit: ', total_score.item())


# In[22]:


clean_text = "1 2 3"
corr_text = "5 3 9"
heads_not_ablate = []  # ablate all heads but not MLPs
mlps_not_ablate = []  # ablate all MLPs
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# # 1 2 3 genr ablation expms

# In[23]:


clean_text = "1 2 3"
corr_text = "5 3 9"


# ## ablate just head 9.1 and MLP 9

# In[24]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## ablate 4.4, 7.11, 9.1

# In[25]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## ablate mlp 9

# In[26]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## ablate 4.4, 7.11, 9.1 and mlp 9

# In[27]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## 6.2, 4.1, 7.1

# In[28]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(6, 2), (4,1), (7,1)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# In[29]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
# heads_not_ablate = [(9, 1)]
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]
len(heads_not_ablate)


# # M T, two days after is

# In[30]:


clean_text = "What comes after Monday is Tuesday, and two days after is"
corr_text = "What comes after X is Y, and two days after is"


# ## clean

# In[31]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## corrupt the subcircuit

# In[32]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## ablate 4.4, 7.11, 9.1

# In[33]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## corrupt 9.1 and mlp9

# In[34]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## ablate mlp 9

# In[35]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## ablate just 9.1

# In[36]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## ablate random head

# In[37]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((6, 2))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# # test clean prompts

# In[38]:


# clean_text = "1"
# tokens = model.to_tokens(clean_text).to(device)
# tokens = tokens[:, 1:] # get rid of prepend bos when using model.to_tokens
# # tokens = model.tokenizer(clean_text)['input_ids']
# logits = model(tokens)
# next_token = logits[0, -1].argmax(dim=-1) # Get the predicted token at the end of our sequence
# next_char = model.to_string(next_token)

clean_text = "1"
corr_ans_tokLen = 3

heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
mlps_not_ablate = [layer for layer in range(12)]
clean_gen(model, clean_text, corr_ans)


# In[39]:


clean_text = "two"
corr_ans_tokLen = 3

heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
mlps_not_ablate = [layer for layer in range(12)]
clean_gen(model, clean_text, corr_ans)


# In[40]:


clean_text = "March"
corr_ans_tokLen = 3

heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
mlps_not_ablate = [layer for layer in range(12)]
clean_gen(model, clean_text, corr_ans)


# In[41]:


clean_text = "Bob is first. David is"
corr_ans_tokLen = 3

heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
mlps_not_ablate = [layer for layer in range(12)]
clean_gen(model, clean_text, corr_ans)


# In[42]:


clean_text = "Two days after Monday is"
corr_ans_tokLen = 3

heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
mlps_not_ablate = [layer for layer in range(12)]
clean_gen(model, clean_text, corr_ans)


# In[43]:


clean_text = "Bob is first in line. David is"
corr_ans_tokLen = 3

heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
mlps_not_ablate = [layer for layer in range(12)]
clean_gen(model, clean_text, corr_ans)


# In[44]:


clean_text = "uno dos tres"
corr_ans_tokLen = 3

heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
mlps_not_ablate = [layer for layer in range(12)]
clean_gen(model, clean_text, corr_ans)


# In[45]:


clean_text = "uno dos tres"
corr_ans_tokLen = 3

heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
mlps_not_ablate = [layer for layer in range(12)]
clean_gen(model, clean_text, corr_ans)


# # Bob is first. David is

# In[46]:


clean_text = "Bob is first. David is"
corr_text = "Bob is X. David is"
corr_ans_tokLen = 1


# clean

# In[47]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt the subcircuit

# In[48]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate 4.4, 7.11, 9.1

# In[49]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt 9.1 and mlp9

# In[50]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate mlp 9

# In[51]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate just 9.1

# In[52]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate random head

# In[53]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((6, 2))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate all

# In[54]:


heads_not_ablate = [ ]

mlps_not_ablate = []

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# # one two three

# In[55]:


clean_text = "one two three"
corr_text = "five nine two"
corr_ans_tokLen = 1


# clean

# In[56]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt the subcircuit

# In[57]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate 4.4, 7.11, 9.1

# In[58]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt 9.1 and mlp9

# In[59]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate mlp 9

# In[60]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate just 9.1

# In[61]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate random head

# In[62]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((6, 2))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate all

# In[63]:


heads_not_ablate = [ ]

mlps_not_ablate = []

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# # January February March

# In[64]:


clean_text = "January February March"
corr_text = "April July July"
corr_ans_tokLen = 1


# clean

# In[65]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt the subcircuit

# In[66]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate 4.4, 7.11, 9.1

# In[67]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt 9.1 and mlp9

# In[68]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate mlp 9

# In[69]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate just 9.1

# In[70]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate random head

# In[71]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((6, 2))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate all

# In[72]:


heads_not_ablate = [ ]

mlps_not_ablate = []

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# # 1 2 3 4 5 6

# In[73]:


clean_text = "1 2 3 4 5 6"
corr_text = "8 5 9 4 2 4"
corr_ans_tokLen = 1


# clean

# In[74]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt the subcircuit

# In[75]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate 4.4, 7.11, 9.1

# In[76]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt 9.1 and mlp9

# In[77]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate mlp 9

# In[78]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate just 9.1

# In[79]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate random head

# In[80]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((6, 2))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate all

# In[81]:


heads_not_ablate = [ ]

mlps_not_ablate = []

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# # 1 to 50

# In[82]:


import random

# Generate a string of numbers from 1 to 50
sequence_string = ' '.join(map(str, range(1, 51)))

# Generate a string of random numbers picked from 1 to 50
random_numbers = [random.randint(1, 50) for _ in range(50)]
random_string = ' '.join(map(str, random_numbers))

print("Sequence String: ", sequence_string)
print("Random Numbers String: ", random_string)


# In[83]:


clean_text = sequence_string
corr_text = random_string
corr_ans_tokLen = 1


# clean

# In[84]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt the subcircuit

# In[85]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate 4.4, 7.11, 9.1

# In[86]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt 9.1 and mlp9

# In[87]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate mlp 9

# In[88]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate just 9.1

# In[89]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate random head

# In[90]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((6, 2))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate all

# In[91]:


heads_not_ablate = [ ]

mlps_not_ablate = []

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# # 1 to 20

# In[108]:


import random

# Generate a string of numbers from 1 to 50
sequence_string = ' '.join(map(str, range(1, 21)))

# Generate a string of random numbers picked from 1 to 50
random_numbers = [random.randint(1, 20) for _ in range(20)]
random_string = ' '.join(map(str, random_numbers))

print("Sequence String: ", sequence_string)
print("Random Numbers String: ", random_string)


# In[109]:


clean_text = sequence_string
corr_text = random_string
corr_ans_tokLen = 1


# clean

# In[110]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt the subcircuit

# In[111]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate 4.4, 7.11, 9.1

# In[112]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt 9.1 and mlp9

# In[113]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate mlp 9

# In[114]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate just 9.1

# In[115]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate random head

# In[116]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((6, 2))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate all

# In[117]:


heads_not_ablate = [ ]

mlps_not_ablate = []

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# # 1 to 100

# In[118]:


import random

# Generate a string of numbers from 1 to 50
sequence_string = ' '.join(map(str, range(1, 101)))

# Generate a string of random numbers picked from 1 to 50
random_numbers = [random.randint(1, 100) for _ in range(100)]
random_string = ' '.join(map(str, random_numbers))

print("Sequence String: ", sequence_string)
print("Random Numbers String: ", random_string)


# In[119]:


clean_text = sequence_string
corr_text = random_string
corr_ans_tokLen = 1


# clean

# In[120]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt the subcircuit

# In[121]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate 4.4, 7.11, 9.1

# In[122]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
head_to_remove = ([(9, 1), (4,4), (7,11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# corrupt 9.1 and mlp9

# In[123]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate mlp 9

# In[124]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated

mlps_not_ablate = [layer for layer in range(12) if layer != 9]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate just 9.1

# In[125]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((9, 1))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate random head

# In[126]:


heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
heads_not_ablate.remove((6, 2))

mlps_not_ablate = [layer for layer in range(12)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# ablate all

# In[127]:


heads_not_ablate = [ ]

mlps_not_ablate = []

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


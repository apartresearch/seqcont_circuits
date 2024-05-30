#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


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
from transformer_lens import HookedTransformer #, HookedTransformerConfig, FactoredMatrix, ActivationCache


# We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.

# In[5]:


torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[6]:


import pdb


# ## Import functions from repo

# In[7]:


get_ipython().system('git clone https://github.com/apartresearch/seqcont_circuits.git')
get_ipython().run_line_magic('cd', '/content/seqcont_circuits/src/iter_node_pruning')


# In[8]:


## comment this out when debugging functions in colab to use funcs defined in colab

# don't improt this
# # from dataset import Dataset

from metrics import *
from head_ablation_fns import *
from mlp_ablation_fns import *
from node_ablation_fns import *
from loop_node_ablation_fns import *


# ## fns

# In[9]:


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
        # word_idx is a dict whose values are tensor with an element for each prompt. The element is the targ token's ind at that prompt
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


# In[10]:


def generate_prompts_list_longer(text, tokens):
    prompts_list = []
    prompt_dict = {
        'corr': str(1),
        'incorr': str(2),
        'text': text
        # 'text': model.to_string(tokens)[0]
        }
    tokens_as_strs = model.tokenizer.tokenize(text)
    # tokens_as_strs = model.to_string(tokens)[0].split()
    # for i in range(tokens.shape[1]):
    for i, tok in enumerate(tokens_as_strs):
        prompt_dict['S'+str(i)] = tok
    # for i, tok in enumerate(tokens):
    #     prompt_dict['S'+str(i)] = model.to_string(tok)

    # prompt_dict = {
    #     'corr': '4',
    #     'incorr': '3',
    #     'text': model.to_string(tokens)[0]
    # }
    # # list_tokens = tokenizer.tokenize('1 2 3 ')
    # tokens_as_strs = model.to_string(tokens)[0].split()
    # for i, tok_as_str in enumerate(tokens_as_strs):
    #     if tok_as_str == '▁':
    #         prompt_dict['S'+str(i)] = ' '
    #     else:
    #         prompt_dict['S'+str(i)] = tok_as_str
    prompts_list.append(prompt_dict)
    return prompts_list


# # Load Model

# In[11]:


from transformers import LlamaForCausalLM, LlamaTokenizer


# In[12]:


get_ipython().system('huggingface-cli login')


# In[13]:


LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH)
# tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH, use_fast= False, add_prefix_space= False)
hf_model = LlamaForCausalLM.from_pretrained(LLAMA_2_7B_CHAT_PATH, low_cpu_mem_usage=True)


# In[14]:


import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer


# In[15]:


model = HookedTransformer.from_pretrained(
    LLAMA_2_7B_CHAT_PATH,
    hf_model = hf_model,
    tokenizer = tokenizer,
    device = "cpu",
    fold_ln = False,
    center_writing_weights = False,
    center_unembed = False,
)

del hf_model

model = model.to("cuda" if torch.cuda.is_available() else "cpu")


# # Load datasets

# In[21]:


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


# In[22]:


pos_dict = {}
for i in range(len(model.tokenizer.tokenize(prompts_list[0]['text']))):
    pos_dict['S'+str(i)] = i


# In[23]:


dataset = Dataset(prompts_list, pos_dict, model.tokenizer)


# In[ ]:


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


# In[ ]:


dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)


# ## Get orig score

# In[ ]:


model.reset_hooks(including_permanent=True)
logits_original = model(dataset.toks)
orig_score = get_logit_diff(logits_original, dataset)
orig_score


# In[ ]:


import gc

del(logits_original)
torch.cuda.empty_cache()
gc.collect()


# # new dataset

# In[16]:


class Dataset:
    def __init__(self, prompts, pos_dict, tokenizer, tokens):  # , S1_is_first=False
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
        # self.toks = torch.Tensor(self.tokenizer(texts, padding=True).input_ids).type(
        #     torch.int
        # )
        self.toks = tokens
        self.corr_tokenIDs = [
            # self.tokenizer.encode(" " + prompt["corr"])[0] for prompt in self.prompts
            self.tokenizer.encode(prompt["corr"])[0] for prompt in self.prompts
        ]
        self.incorr_tokenIDs = [
            # self.tokenizer.encode(" " + prompt["incorr"])[0] for prompt in self.prompts
            self.tokenizer.encode(prompt["incorr"])[0] for prompt in self.prompts
        ]

        # word_idx: for every prompt, find the token index of each target token and "end"
        # word_idx is a dict whose values are tensor with an element for each prompt. The element is the targ token's ind at that prompt
        self.word_idx = {}
        # for targ in [key for key in self.prompts[0].keys() if (key != 'text' and key != 'corr' and key != 'incorr')]:
        for targ in [key for key in pos_dict]:
            targ_lst = []
            for prompt in self.prompts:
                input_text = prompt["text"]
                # tokens = self.tokenizer.tokenize(input_text)
                # if S1_is_first and targ == "S1":  # only use this if first token doesn't have space Ġ in front
                #     target_token = prompt[targ]
                # else:
                #     target_token = "Ġ" + prompt[targ]
                # target_index = tokens.index(target_token)
                target_index = pos_dict[targ]
                targ_lst.append(target_index)
            self.word_idx[targ] = torch.tensor(targ_lst)

        # targ_lst = []
        # for prompt in self.prompts:
        #     input_text = prompt["text"]
        #     tokens = self.tokenizer.tokenize(input_text)
        #     end_token_index = len(tokens) - 1
        #     targ_lst.append(end_token_index)
        # self.word_idx["end"] = torch.tensor(targ_lst)

    def __len__(self):
        return self.N


# # logit diff for mult tok answers

# In[24]:


def clean_gen(model, clean_text, corr_ans):
    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
    tokens = model.to_tokens(clean_text).to(device)
    # tokens = tokens[:, 1:] # get rid of prepend bos when using model.to_tokens

    total_score = 0
    corr_ans_tokLen = 0
    ans_so_far = ''
    # while True:
    for i in range(5):
        print(f"Sequence so far: {model.to_string(tokens)[0]!r}")
        logits = model(tokens)
        next_token = logits[0, -1].argmax(dim=-1) # Get the predicted token at the end of our sequence
        next_char = model.to_string(next_token)

        # dataset = Dataset(prompts_list, pos_dict, model.tokenizer)
        dataset = Dataset(prompts_list, pos_dict, model.tokenizer, tokens)

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
        # if next_char == '':
        #     next_char = ' '
        # clean_text = clean_text + next_char
        # tokens = model.to_tokens(clean_text).to(device)
    return corr_ans_tokLen


# In[25]:


clean_text = "1 2 3"
corr_ans = ' 5'
corr_ans_tokLen = clean_gen(model, clean_text, corr_ans)


# In[26]:


def ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen):
    tokens = model.to_tokens(clean_text).to(device)
    prompts_list = generate_prompts_list_longer(clean_text, tokens)

    corr_tokens = model.to_tokens(corr_text).to(device)
    prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)

    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
    pos_dict = {}
    # for i in range(len(model.tokenizer.tokenize(prompts_list_2[0]['text']))):
    num_pos = len(model.tokenizer(prompts_list_2[0]['text']).input_ids)
    # for i in range(num_pos + 1):
    for i in range(num_pos ):
        pos_dict['S'+str(i)] = i
    # dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)
    dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer, tokens)
    # pdb.set_trace()
    model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

    # tokens = tokens[:, 1:] # get rid of prepend bos when using model.to_tokens
    logits = model(tokens)
    next_token = logits[0, -1].argmax(dim=-1)
    next_char = model.to_string(next_token)

    total_score = 0

    for i in range(corr_ans_tokLen):
    # for i in range(5):
        if next_char == '':
            next_char = ' '
        # print(f"Sequence so far: {model.to_string(tokens)[0]!r}")
        # print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

        clean_text = clean_text + next_char
        # tokens = model.to_tokens(clean_text).to(device)
        # tokens = tokens[:, 1:]
        # print(clean_text)
        print(f"Sequence so far: {clean_text}")
        print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

        # clean_text = model.to_string(tokens)[0]
        tokens = torch.cat([tokens, next_token[None, None]], dim=-1)
        print(clean_text)
        # print(tokens.shape)

        # get new ablation dataset
        model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

        corr_text = corr_text + next_char
        # corr_tokens = model.to_tokens(corr_text).to(device)

        # corr_text = model.to_string(corr_tokens)[0]
        corr_tokens = torch.cat([corr_tokens, next_token[None, None]], dim=-1)
        prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)
        print(corr_text)
        # print(corr_tokens.shape)

        pos_dict = {}
        # for i in range(len(model.tokenizer.tokenize(prompts_list_2[0]['text']))):
        # for i in range(corr_tokens.shape[1]):
        num_pos = len(model.tokenizer(prompts_list_2[0]['text']).input_ids)
        # for i in range(num_pos + 1):
        for i in range(num_pos ):
            pos_dict['S'+str(i)] = i

        # dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)
        dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer, corr_tokens)

        model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

        logits = model(tokens)
        next_token = logits[0, -1].argmax(dim=-1) # Get the predicted token at the end of our sequence
        next_char = model.to_string(next_token)

        # print('\n')
        # print(f"Sequence so far: {model.to_string(tokens)[0]!r}")

        # new_score = get_logit_diff(logits, dataset)
        # total_score += new_score
        # print(f"corr logit of new char: {new_score}")
    # print('\n Total corr logit: ', total_score.item())


# # test

# In[ ]:


# heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
# mlps_not_ablate = [layer for layer in range(32)]
# clean_text = "1 2 3"
# tokens = model.to_tokens(clean_text).to(device)
# prompts_list = generate_prompts_list_longer(clean_text, tokens)
# corr_text = "5 3 9"
# corr_tokens = model.to_tokens(corr_text).to(device)
# prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)


# In[ ]:


# tokens = model.to_tokens(clean_text).to(device)
# prompts_list = generate_prompts_list_longer(clean_text, tokens)

# corr_tokens = model.to_tokens(corr_text).to(device)
# prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)

# model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
# pos_dict = {}
# for i in range(len(model.tokenizer.tokenize(prompts_list_2[0]['text']))):
#     pos_dict['S'+str(i)] = i
# dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)
# model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

# # tokens = tokens[:, 1:] # get rid of prepend bos when using model.to_tokens
# # pdb.set_trace()
# logits = model(tokens)
# next_token = logits[0, -1].argmax(dim=-1)
# next_char = model.to_string(next_token)

# total_score = 0

# for i in range(1):
# # for i in range(5):
#     if next_char == '':
#         next_char = ' '
#     print(f"Sequence so far: {model.to_string(tokens)[0]!r}")
#     print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

#     # if next_char == '':
#     #     clean_text = clean_text + ' '
#     # else:
#     clean_text = clean_text + next_char
#     # tokens = model.to_tokens(clean_text).to(device)
#     # tokens = tokens[:, 1:]
#     # print(clean_text)

#     # clean_text = model.to_string(tokens)[0]
#     tokens = torch.cat([tokens, next_token[None, None]], dim=-1)
#     print(clean_text)
#     # print(tokens.shape)

#     # get new ablation dataset
#     model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

#     corr_text = corr_text + next_char
#     # corr_tokens = model.to_tokens(corr_text).to(device)

#     # corr_text = model.to_string(corr_tokens)[0]
#     corr_tokens = torch.cat([corr_tokens, next_token[None, None]], dim=-1)
#     prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)
#     print(corr_text)
#     # print(corr_tokens.shape)

#     pos_dict = {}
#     # for i in range(len(model.tokenizer.tokenize(prompts_list_2[0]['text']))):
#     for i in range(corr_tokens.shape[1]):
#         pos_dict['S'+str(i)] = i

#     dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)

#     # model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

#     logits = model(tokens)
#     next_token = logits[0, -1].argmax(dim=-1) # Get the predicted token at the end of our sequence
#     next_char = model.to_string(next_token)

#     # print('\n')
#     # print(f"Sequence so far: {model.to_string(tokens)[0]!r}")

#     # new_score = get_logit_diff(logits, dataset)
#     # total_score += new_score
#     # print(f"corr logit of new char: {new_score}")
# # print('\n Total corr logit: ', total_score.item())


# # test clean prompts

# In[ ]:


# clean_text = "1"
# tokens = model.to_tokens(clean_text).to(device)
# tokens = tokens[:, 1:] # get rid of prepend bos when using model.to_tokens
# # tokens = model.tokenizer(clean_text)['input_ids']
# logits = model(tokens)
# next_token = logits[0, -1].argmax(dim=-1) # Get the predicted token at the end of our sequence
# next_char = model.to_string(next_token)

clean_text = "1"
corr_ans_tokLen = 3
clean_gen(model, clean_text, corr_ans)


# In[ ]:


clean_text = "two"
corr_ans_tokLen = 3
clean_gen(model, clean_text, corr_ans)


# In[ ]:


clean_text = "March"
corr_ans_tokLen = 3
clean_gen(model, clean_text, corr_ans)


# In[ ]:


clean_text = "Bob is first. David is"
corr_ans_tokLen = 3
clean_gen(model, clean_text, corr_ans)


# In[ ]:


clean_text = "Two days after Monday is"
corr_ans_tokLen = 3
clean_gen(model, clean_text, corr_ans)


# In[ ]:


clean_text = "uno dos tres"
corr_ans_tokLen = 3
clean_gen(model, clean_text, corr_ans)


# In[ ]:


clean_text = "one two three"
corr_ans_tokLen = 3
clean_gen(model, clean_text, corr_ans)


# In[ ]:


clean_text = "2 4 6 "
corr_ans_tokLen = 3
clean_gen(model, clean_text, corr_ans)


# # test if diff answers in two genr ways

# In[ ]:


# Get list of arguments to pass to `generate` (specifically these are the ones relating to sampling)
generate_kwargs = dict(
    do_sample = False, # deterministic output so we can compare it to the HF model
    top_p = 1.0, # suppresses annoying output errors
    temperature = 1.0, # suppresses annoying output errors
)


# In[ ]:


prompt = "uno dos tres"
output = model.generate(prompt, max_new_tokens=1, **generate_kwargs)
print(output)


# In[ ]:


prompt = "one two three"
output = model.generate(prompt, max_new_tokens=1, **generate_kwargs)
print(output)


# # 1 2 3 genr ablation expms

# In[29]:


def mask_circ_heads(
    means_dataset: Dataset,
    model: HookedTransformer,
    circuit: Dict[str, List[Tuple[int, int]]],
    seq_pos_to_keep: Dict[str, str],
) -> Dict[int, Bool[Tensor, "batch seq head"]]:
    '''
    Output: for each layer, a mask of circuit components that should not be ablated
    '''
    heads_and_posns_to_keep = {}
    batch, seq, n_heads = len(means_dataset), means_dataset.max_len, model.cfg.n_heads

    for layer in range(model.cfg.n_layers):

        mask = t.zeros(size=(batch, seq, n_heads))

        for (head_type, head_list) in circuit.items():
            seq_pos = seq_pos_to_keep[head_type]
            # if seq_pos == 'S7':
            #     pdb.set_trace()
            indices = means_dataset.word_idx[seq_pos] # modify this for key vs query pos. curr, this is query
            for (layer_idx, head_idx) in head_list:
                if layer_idx == layer:
                    # if indices.item() == 7:
                    #     pdb.set_trace()
                    mask[:, indices, head_idx] = 1
                    # mask[:, :, head_idx] = 1  # keep L.H at all pos

        heads_and_posns_to_keep[layer] = mask.bool()
    # pdb.set_trace()
    return heads_and_posns_to_keep


# In[27]:


# from dataset import Dataset
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
import einops
from functools import partial
import torch as t
from torch import Tensor
from typing import Dict, Tuple, List
from jaxtyping import Float, Bool

# from head_ablation_fns import *
# from mlp_ablation_fns import *

def add_ablation_hook_MLP_head(
    model: HookedTransformer,
    means_dataset: Dataset,
    heads_lst, mlp_lst,
    is_permanent: bool = True,
) -> HookedTransformer:
    CIRCUIT = {}
    SEQ_POS_TO_KEEP = {}
    # for i in range(len(model.tokenizer.tokenize(means_dataset.prompts[0]['text']))):
    num_pos = len(model.tokenizer(means_dataset.prompts[0]['text']).input_ids)
    for i in range(num_pos ):
        CIRCUIT['S'+str(i)] = heads_lst
        # if i == len(model.tokenizer.tokenize(means_dataset.prompts[0]['text'])) - 1:
        # if i == num_pos - 1:
        #     SEQ_POS_TO_KEEP['S'+str(i)] = 'end'
        # else:
        SEQ_POS_TO_KEEP['S'+str(i)] = 'S'+str(i)

    model.reset_hooks(including_permanent=True)

    # Compute the mean of each head's output on the ABC dataset, grouped by template
    means = get_heads_actv_mean(means_dataset, model)
    # Convert this into a boolean map
    components_to_keep = mask_circ_heads(means_dataset, model, CIRCUIT, SEQ_POS_TO_KEEP)

    # Get a hook function which will patch in the mean z values for each head, at
    # all positions which aren't important for the circuit
    hook_fn = partial(
        hook_func_mask_head,
        # components_to_keep=components_to_keep,
        # means=means,
        circuit=CIRCUIT,
    )

    # Apply hook
    model.add_hook(lambda name: name.endswith("z"), hook_fn, is_permanent=is_permanent)

    # if all_entries_true(components_to_keep) == False:
    #     pdb.set_trace()
    ########################
    # CIRCUIT = {}
    # SEQ_POS_TO_KEEP = {}
    # # for i in range(len(model.tokenizer.tokenize(means_dataset.prompts[0]['text']))):
    # num_pos = len(model.tokenizer(means_dataset.prompts[0]['text']).input_ids)
    # for i in range(num_pos ):
    #     CIRCUIT['S'+str(i)] = mlp_lst
    #     # if i == len(model.tokenizer.tokenize(means_dataset.prompts[0]['text'])) - 1:
    #     # if i == num_pos - 1:
    #     #     SEQ_POS_TO_KEEP['S'+str(i)] = 'end'
    #     # else:
    #     SEQ_POS_TO_KEEP['S'+str(i)] = 'S'+str(i)

    # # Compute the mean of each head's output on the ABC dataset, grouped by template
    # means = get_MLPs_actv_mean(means_dataset, model)

    # # Convert this into a boolean map
    # components_to_keep = mask_circ_MLPs(means_dataset, model, CIRCUIT, SEQ_POS_TO_KEEP)

    # # Get a hook function which will patch in the mean z values for each head, at
    # # all positions which aren't important for the circuit
    # hook_fn = partial(
    #     hook_func_mask_mlp_out,
    #     components_to_keep=components_to_keep,
    #     means=means
    # )

    # Apply hook
    # model.add_hook(lambda name: name.endswith("mlp_out"), hook_fn, is_permanent=True)

    return model


# In[ ]:


clean_text = "1 2 3"
corr_text = "5 3 9"
# corr_text = "1 2 3"


# clean

# In[ ]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[ ]:


def all_entries_true(tensor_dict):
    for key, tensor in tensor_dict.items():
        if not torch.all(tensor).item():
            return False
    return True


# ablate just head 20.7

# In[ ]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate.remove((20, 7))

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[ ]:


clean_text = "1 2 3"
corr_text = "5 3 9"
heads_not_ablate = []  # ablate all heads but not MLPs
mlps_not_ablate = []  # ablate all MLPs
corr_ans_tokLen = 2
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# In[ ]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(20, 7), (1, 11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[ ]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(20, 7), (1, 11)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32) if layer != 1]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[ ]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(20, 7), (1, 11), (16,0)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32) if layer != 1]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[ ]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(20, 7), (1, 11), (16,0)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32) if layer <10]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[ ]:


heads_not_ablate = []

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[ ]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(20, 7), (1, 11), (16,0), (0, 30), (0, 9), (15,25)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# # 2 4 6

# In[31]:


clean_text = "2 4 6"
corr_text = "5 3 9"
# corr_text = "1 2 3"


# clean

# In[32]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[33]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(0, 13)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[34]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(0, 1)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[35]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(1, 14)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# # uno dos tres

# In[28]:


model.to_tokens("uno dos tres")


# In[29]:


model.to_tokens("uno uno uno")


# In[30]:


clean_text = "uno dos tres"
corr_text = "uno uno uno" # dos tres cinco seis


# In[31]:


def get_heads_actv_mean(
    means_dataset: Dataset,
    model: HookedTransformer
) -> Float[Tensor, "layer batch seq head_idx d_head"]:
    '''
    Output: The mean activations of a head's output
    '''
    _, means_cache = model.run_with_cache(
        means_dataset.toks.long(),
        return_type=None,
        names_filter=lambda name: name.endswith("z"),
    )
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    batch, seq_len = len(means_dataset), means_dataset.max_len
    means = t.zeros(size=(n_layers, batch, seq_len, n_heads, d_head), device=model.cfg.device)

    # for layer in range(model.cfg.n_layers):
    #     z_for_this_layer: Float[Tensor, "batch seq head d_head"] = means_cache[utils.get_act_name("z", layer)]
    #     for template_group in means_dataset.groups:
    #         z_for_this_template = z_for_this_layer[template_group]
    #         z_means_for_this_template = einops.reduce(z_for_this_template, "batch seq head d_head -> seq head d_head", "mean")
    #         if z_means_for_this_template.shape[0] == 5:
    #             pdb.set_trace()
    #         means[layer, template_group] = z_means_for_this_template

    del(means_cache)

    return means


# In[32]:


def mask_circ_heads(
    means_dataset: Dataset,
    model: HookedTransformer,
    circuit: Dict[str, List[Tuple[int, int]]],
    seq_pos_to_keep: Dict[str, str],
) -> Dict[int, Bool[Tensor, "batch seq head"]]:
    '''
    Output: for each layer, a mask of circuit components that should not be ablated
    '''
    heads_and_posns_to_keep = {}
    # batch, seq, n_heads = len(means_dataset), means_dataset.max_len, model.cfg.n_heads
    batch, seq, n_heads = len(means_dataset), len(circuit.keys()), model.cfg.n_heads
    print(seq)

    for layer in range(model.cfg.n_layers):

        mask = t.zeros(size=(batch, seq, n_heads))

        for (head_type, head_list) in circuit.items():
            seq_pos = seq_pos_to_keep[head_type]
            indices = means_dataset.word_idx[seq_pos] # modify this for key vs query pos. curr, this is query
            for (layer_idx, head_idx) in head_list:
                if layer_idx == layer:
                    # mask[:, indices, head_idx] = 1
                    mask[:, :, head_idx] = 1

        heads_and_posns_to_keep[layer] = mask.bool()

    return heads_and_posns_to_keep


# In[33]:


def hook_func_mask_head(
    z: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    # components_to_keep: Dict[int, Bool[Tensor, "batch seq head"]],
    # means: Float[Tensor, "layer batch seq head d_head"],
    circuit: Dict[str, List[Tuple[int, int]]],
) -> Float[Tensor, "batch seq head d_head"]:
    '''
    Use this to not mask components
    '''
    # mask_for_this_layer = components_to_keep[hook.layer()].unsqueeze(-1).to(z.device)
    # z = t.where(mask_for_this_layer, z, means[hook.layer()])

    ###
    # heads_and_posns_to_keep = {}
    # batch, seq, n_heads = z.shape[0], z.shape[1], model.cfg.n_heads  # components_to_keep[0].shape[0] is batch

    # for layer in range(model.cfg.n_layers):

    #     mask = t.zeros(size=(batch, seq, n_heads))

    #     for (head_type, head_list) in circuit.items():
    #         # seq_pos = seq_pos_to_keep[head_type]
    #         # indices = means_dataset.word_idx[seq_pos] # modify this for key vs query pos. curr, this is query
    #         for (layer_idx, head_idx) in head_list:
    #             if layer_idx == layer:
    #                 # mask[:, indices, head_idx] = 1
    #                 mask[:, :, head_idx] = 1

    #     heads_and_posns_to_keep[layer] = mask.bool()
    ###
    mask_for_this_layer = t.zeros(size=(z.shape[0], z.shape[1], z.shape[2]))
    for (head_type, head_list) in circuit.items():
        # seq_pos = seq_pos_to_keep[head_type]
        # indices = means_dataset.word_idx[seq_pos] # modify this for key vs query pos. curr, this is query
        for (layer_idx, head_idx) in head_list:
            if layer_idx == hook.layer():
                # mask[:, indices, head_idx] = 1
                mask_for_this_layer[:, :, head_idx] = 1

    mask_for_this_layer = mask_for_this_layer.bool()
    mask_for_this_layer = mask_for_this_layer.unsqueeze(-1).to(z.device)  # d_model is 1; then is broadcast in where

    z = t.where(mask_for_this_layer, z, 0)

    return z


# In[34]:


def add_ablation_hook_head(
    model: HookedTransformer,
    means_dataset: Dataset,
    circuit: Dict[str, List[Tuple[int, int]]],
    seq_pos_to_keep: Dict[str, str],
    is_permanent: bool = True,
) -> HookedTransformer:
    '''
    Ablate the model, except as components and positions to keep
    '''

    model.reset_hooks(including_permanent=True)
    means = get_heads_actv_mean(means_dataset, model)
    components_to_keep = mask_circ_heads(means_dataset, model, circuit, seq_pos_to_keep)

    hook_fn = partial(
        hook_func_mask_head,
        # components_to_keep=components_to_keep,
        # means=means,
        circuit=circuit,
    )

    model.add_hook(lambda name: name.endswith("z"), hook_fn, is_permanent=is_permanent)
    return model


# clean

# In[35]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[36]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(0, 13)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[37]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(20, 17)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# # seqcont word problems

# In[38]:


clean_text = "What comes after Monday is Tuesday, and two days after is"
corr_text = "What comes after X is Y, and two days after is"


# clean

# In[39]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[40]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(0, 13)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[41]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(20, 17)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# Obtain heads from top 20 of https://colab.research.google.com/drive/1p_x98vp4OMx46rphUdIk64E7P94cQmg9#scrollTo=susSZdqpqVzd&line=1&uniqifier=1

# In[42]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = ([(20, 17), (1, 11), (0, 30), (0, 9), (5, 26), (16, 0), (13, 6), (15, 25),
    (5, 15), (6, 11), (5, 25), (5, 17), (1, 28), (29, 5), (4, 3), (15, 15),
    (26, 2), (10, 25), (2, 2), (23, 2)])
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


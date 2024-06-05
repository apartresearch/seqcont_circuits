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


# # new ablation functions

# In[35]:


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


# In[36]:


# def mask_circ_heads(
#     means_dataset: Dataset,
#     model: HookedTransformer,
#     circuit: Dict[str, List[Tuple[int, int]]],
#     seq_pos_to_keep: Dict[str, str],
# ) -> Dict[int, Bool[Tensor, "batch seq head"]]:
#     '''
#     Output: for each layer, a mask of circuit components that should not be ablated
#     '''
#     heads_and_posns_to_keep = {}
#     batch, seq, n_heads = len(means_dataset), means_dataset.max_len, model.cfg.n_heads

#     for layer in range(model.cfg.n_layers):

#         mask = t.zeros(size=(batch, seq, n_heads))

#         for (head_type, head_list) in circuit.items():
#             seq_pos = seq_pos_to_keep[head_type]
#             # if seq_pos == 'S7':
#             #     pdb.set_trace()
#             indices = means_dataset.word_idx[seq_pos] # modify this for key vs query pos. curr, this is query
#             for (layer_idx, head_idx) in head_list:
#                 if layer_idx == layer:
#                     # if indices.item() == 7:
#                     #     pdb.set_trace()
#                     mask[:, indices, head_idx] = 1
#                     # mask[:, :, head_idx] = 1  # keep L.H at all pos

#         heads_and_posns_to_keep[layer] = mask.bool()
#     # pdb.set_trace()
#     return heads_and_posns_to_keep


# In[37]:


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


# In[38]:


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


# In[39]:


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


# In[40]:


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

    # model.add_hook(lambda name: name.endswith("mlp_out"), hook_fn, is_permanent=True)

    return model


# In[41]:


def all_entries_true(tensor_dict):
    for key, tensor in tensor_dict.items():
        if not torch.all(tensor).item():
            return False
    return True


# # ablation fns mult tok answers

# In[33]:


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


# In[34]:


def ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen):
    tokens = model.to_tokens(clean_text).to(device)
    prompts_list = generate_prompts_list_longer(clean_text, tokens)

    corr_tokens = model.to_tokens(corr_text).to(device)
    prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)

    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
    pos_dict = {}
    num_pos = len(model.tokenizer(prompts_list_2[0]['text']).input_ids)
    for i in range(num_pos ):
        pos_dict['S'+str(i)] = i
    dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)
    model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

    logits = model(tokens)
    next_token = logits[0, -1].argmax(dim=-1)
    next_char = model.to_string(next_token)

    total_score = 0

    for i in range(corr_ans_tokLen):
        if next_char == '':
            next_char = ' '

        clean_text = clean_text + next_char
        print(f"Sequence so far: {clean_text}")
        print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

        tokens = torch.cat([tokens, next_token[None, None]], dim=-1)

        # get new ablation dataset
        # model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

        # corr_text = corr_text + next_char
        # corr_tokens = torch.cat([corr_tokens, next_token[None, None]], dim=-1)
        # prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)

        # pos_dict = {}
        # num_pos = len(model.tokenizer(prompts_list_2[0]['text']).input_ids)
        # for i in range(num_pos ):
        #     pos_dict['S'+str(i)] = i

        # dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer, corr_tokens)

        # model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

        logits = model(tokens)
        next_token = logits[0, -1].argmax(dim=-1) # Get the predicted token at the end of our sequence
        next_char = model.to_string(next_token)

        # new_score = get_logit_diff(logits, dataset)
        # total_score += new_score
        # print(f"corr logit of new char: {new_score}")
    # print('\n Total corr logit: ', total_score.item())


# # alt ablate then gen fns

# In[20]:


# Get list of arguments to pass to `generate` (specifically these are the ones relating to sampling)
generate_kwargs = dict(
    do_sample = False, # deterministic output so we can compare it to the HF model
    top_p = 1.0, # suppresses annoying output errors
    temperature = 1.0, # suppresses annoying output errors
)


# In[21]:


def ablate_then_gen_circ(model, generate_kwargs, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen):
    corr_tokens = model.to_tokens(corr_text).to(device)
    prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)

    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
    pos_dict = {}
    num_pos = len(model.tokenizer(prompts_list_2[0]['text']).input_ids)
    for i in range(num_pos ):
        pos_dict['S'+str(i)] = i
    dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)
    model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

    output = model.generate(clean_text, max_new_tokens= corr_ans_tokLen, **generate_kwargs)
    print(output)


# ## test

# In[ ]:


prompt =  "January February March"
model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
output = model.generate(prompt, max_new_tokens=1, **generate_kwargs)
print(output)


# In[ ]:


# tokens = model.to_tokens(clean_text).to(device)
# prompts_list = generate_prompts_list_longer(clean_text, tokens)

corr_text = "9 6 5"
corr_tokens = model.to_tokens(corr_text).to(device)
prompts_list_2 = generate_prompts_list_longer(corr_text, corr_tokens)

model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
pos_dict = {}
num_pos = len(model.tokenizer(prompts_list_2[0]['text']).input_ids)
for i in range(num_pos ):
    pos_dict['S'+str(i)] = i
dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)
heads_not_ablate = []
mlps_not_ablate = []
model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)


# In[ ]:


prompt =  "January February March"
output = model.generate(prompt, max_new_tokens=1, **generate_kwargs)
print(output)


# In[ ]:


prompt =  "January February March"
output = model.generate(prompt, max_new_tokens=2, **generate_kwargs)
print(output)


# In[28]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
corr_ans_tokLen = 1
# ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)
ablate_then_gen_circ(model, generate_kwargs, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, corr_ans_tokLen)


# # Define circs

# In[45]:


# nw_circ = [(0, 1), (0, 4), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (1, 16), (1, 24), (1, 27), (1, 28), (2, 2), (2, 5), (2, 8), (2, 24), (2, 30), (3, 7), (3, 14), (3, 19), (3, 23), (4, 3), (5, 16), (5, 25), (6, 11), (6, 14), (7, 0), (7, 30), (8, 0), (8, 2), (8, 3), (8, 4), (8, 6), (8, 21), (8, 31), (9, 1), (9, 3), (9, 7), (9, 11), (9, 29), (9, 31), (10, 13), (10, 18), (10, 23), (10, 24), (10, 25), (10, 27), (11, 18), (11, 28), (12, 18), (12, 26), (13, 11), (13, 17), (13, 18), (13, 19), (13, 20), (13, 21), (13, 23), (14, 7), (14, 14), (15, 25), (15, 28), (16, 0), (16, 12), (16, 14), (16, 15), (16, 16), (16, 19), (16, 24), (16, 29), (17, 17), (17, 23), (17, 31), (18, 31), (19, 12), (20, 17), (27, 20), (27, 25), (27, 27), (27, 31), (28, 5), (29, 5)]
# in order from most impt to least based on how much changes perf when ablated
nw_circ = [(20, 17), (5, 25), (16, 0), (29, 5), (3, 19), (6, 11), (15, 25), (8, 0), (16, 24), (8, 4), (7, 0), (6, 14), (16, 29), (5, 16), (12, 26), (4, 3), (3, 7), (7, 30), (11, 28), (28, 5), (17, 31), (13, 11), (13, 20), (12, 18), (1, 27), (10, 13), (18, 31), (8, 6), (9, 1), (0, 4), (2, 2), (9, 11), (19, 12), (1, 16), (13, 17), (9, 7), (11, 18), (2, 24), (10, 18), (9, 31), (9, 29), (2, 30), (2, 5), (1, 24), (2, 8), (15, 28), (27, 31), (16, 14), (3, 23), (3, 14), (10, 23), (27, 20), (8, 3), (14, 7), (14, 14), (16, 15), (8, 2), (17, 17), (0, 1), (10, 27), (16, 19), (0, 8), (0, 12), (1, 28), (0, 11), (17, 23), (0, 10), (0, 6), (13, 19), (8, 31), (10, 24), (16, 12), (13, 23), (13, 21), (27, 27), (9, 3), (27, 25), (16, 16), (8, 21), (0, 7), (13, 18), (10, 25)]


# In[48]:


impt_months_heads = ([(23, 17), (17, 11), (16, 0), (26, 14), (18, 9), (5, 25), (22, 20), (6, 24), (26, 9), (12, 18), (13, 20), (19, 12), (27, 29), (13, 14), (16, 14), (12, 26), (19, 30), (16, 18), (31, 27), (26, 28), (16, 1), (18, 1), (19, 28), (18, 31), (29, 4), (17, 0), (14, 1), (17, 12), (12, 15), (28, 16), (10, 1), (16, 19), (9, 27), (30, 1), (19, 27), (0, 3), (15, 11), (21, 3), (11, 19), (12, 0), (23, 11), (8, 14), (16, 8), (22, 13), (13, 3), (4, 19), (14, 15), (12, 20), (19, 16), (18, 5)])


# # test prompts

# In[87]:


clean_text = "What comes after the first step in a process? Be concise."
# corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 6)


# In[88]:


clean_text = "What comes after the second item in a list? Be concise."
# corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 30)


# In[ ]:


clean_text = "What comes after the second item in a list?"
# corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 30)


# In[108]:


clean_text = "If today is the 14th of a month, what date will it be in 10 days?"
# corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 30)


# # 1 2 3 genr ablation expms

# In[59]:


clean_text = "1 2 3"
corr_text = "5 3 9"
# corr_text = "1 2 3"


# clean

# In[42]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
corr_ans_tokLen = 3
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# ## ablate circ

# In[44]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# ## top nw circ heads

# In[61]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ[:10]
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 6)


# In[62]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ[:20]
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 6)


# ## top nwcirc + impt month heads

# In[49]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove_num = nw_circ[:10]
head_to_remove_months = impt_months_heads[:20]
head_to_remove = head_to_remove_num + head_to_remove_months
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 10)


# In[50]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove_num = nw_circ[:20]
head_to_remove_months = impt_months_heads[:20]
head_to_remove = head_to_remove_num + head_to_remove_months
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 10)


# # 2 4 6

# In[65]:


clean_text = "2 4 6"
corr_text = "5 3 9"
# corr_text = "1 2 3"


# ## ablate circ

# In[52]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# ## top nw circ heads

# In[66]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ[:10]
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# In[67]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ[:15]
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 6)


# In[68]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ[:20]
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 6)


# ## top nwcirc + impt month heads

# In[54]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove_num = nw_circ[:10]
head_to_remove_months = impt_months_heads[:20]
head_to_remove = head_to_remove_num + head_to_remove_months
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 10)


# In[55]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove_num = nw_circ[:20]
head_to_remove_months = impt_months_heads[:20]
head_to_remove = head_to_remove_num + head_to_remove_months
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 10)


# ## random

# In[56]:


import random

# Generate all possible pairs (i, j) where i and j range from 0 to 31 inclusive
all_possible_pairs = [(i, j) for i in range(32) for j in range(32)]

# Randomly choose 100 pairs from the list
head_to_remove = random.sample(all_possible_pairs, 50)

heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# # uno dos tres

# In[69]:


clean_text = "uno dos tres"
corr_text = "uno uno uno" # dos tres cinco seis


# clean

# In[ ]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 10)


# ## ablate circ

# In[71]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# ## top nw circ heads

# In[70]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ[:10]
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# ## random 1

# In[ ]:


import random

# Generate all possible pairs (i, j) where i and j range from 0 to 31 inclusive
all_possible_pairs = [(i, j) for i in range(32) for j in range(32)]

# Randomly choose 100 pairs from the list
head_to_remove = random.sample(all_possible_pairs, 1)

heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 6)


# In[ ]:


import random

# Generate all possible pairs (i, j) where i and j range from 0 to 31 inclusive
all_possible_pairs = [(i, j) for i in range(32) for j in range(32)]

# Randomly choose 100 pairs from the list
head_to_remove = random.sample(all_possible_pairs, 1)

heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 6)


# ## random 10

# In[78]:


# Function to randomly choose 50 pairs ensuring less than 10 overlap with heads_of_circ
def choose_heads_to_remove(filtered_pairs, heads_of_circ, num_pairs=50, max_overlap=10):
    while True:
        head_to_remove = random.sample(filtered_pairs, num_pairs)
        overlap_count = len([head for head in head_to_remove if head in heads_of_circ])
        if overlap_count < max_overlap:
            return head_to_remove


# In[80]:


import random

heads_of_circ = nw_circ

all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
# Filter out heads_of_circ from all_possible_pairs
filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_of_circ]

# Randomly choose 10 pairs ensuring less than 10 overlaps with heads_of_circ
head_to_remove = choose_heads_to_remove(filtered_pairs, heads_of_circ, 10, 10)

heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]

mlps_not_ablate = [layer for layer in range(32)]

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 6)


# In[75]:


import random

# Generate all possible pairs (i, j) where i and j range from 0 to 31 inclusive
all_possible_pairs = [(i, j) for i in range(32) for j in range(32)]

# Randomly choose 100 pairs from the list
head_to_remove = random.sample(all_possible_pairs, 10)

heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)] #  if layer <10

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 6)


# In[76]:


head_to_remove


# In[77]:


top50_1234 = ([(20, 17), (1, 11), (0, 30), (0, 9), (5, 26), (16, 0), (13, 6), (15, 25), (5, 15), (6, 11), (5, 25), (5, 17), (1, 28), (29, 5), (4, 3), (15, 15), (26, 2), (10, 25), (2, 2), (23, 2), (30, 13), (25, 23), (6, 24), (11, 28), (10, 1), (7, 0), (18, 19), (0, 26), (9, 26), (18, 9), (19, 28), (10, 5), (2, 24), (15, 26), (31, 4), (1, 16), (11, 27), (31, 11), (16, 14), (23, 17), (23, 27), (14, 8), (1, 22), (23, 6), (7, 30), (19, 30), (3, 6), (19, 12), (20, 25), (17, 17)])
overlap = list(set(top50_1234) & set(head_to_remove))
overlap


# # "What comes after the second item in a list? Be concise.”

# In[90]:


clean_text = "What comes after the second item in a list? Be concise."
# corr_text = "uno uno uno" # dos tres cinco seis
num_toks_gen = 10


# clean

# In[91]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
num_toks_gen = 6
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# ## ablate circ

# In[94]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]
num_toks_gen = 30

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# ## top nw circ heads

# In[95]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ[:10]
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

num_toks_gen = 30
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# ## random 10

# In[96]:


# Function to randomly choose 50 pairs ensuring less than 10 overlap with heads_of_circ
def choose_heads_to_remove(filtered_pairs, heads_of_circ, num_pairs=50, max_overlap=10):
    while True:
        head_to_remove = random.sample(filtered_pairs, num_pairs)
        overlap_count = len([head for head in head_to_remove if head in heads_of_circ])
        if overlap_count < max_overlap:
            return head_to_remove


# In[97]:


import random

heads_of_circ = nw_circ

all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
# Filter out heads_of_circ from all_possible_pairs
filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_of_circ]

# Randomly choose 10 pairs ensuring less than 10 overlaps with heads_of_circ
head_to_remove = choose_heads_to_remove(filtered_pairs, heads_of_circ, 10, 10)

heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]

mlps_not_ablate = [layer for layer in range(32)]

num_toks_gen = 30
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# # "What comes after the second item in a list? The next item in a list is the”

# In[102]:


clean_text = "What comes after the second item in a list? The next item in a list is the"
# corr_text = "uno uno uno" # dos tres cinco seis
num_toks_gen = 10


# clean

# In[103]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
num_toks_gen = 6
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# ## ablate circ

# In[104]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]
num_toks_gen = 30

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# ## top nw circ heads

# In[105]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ[:10]
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

num_toks_gen = 30
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# ## random 10

# In[106]:


# Function to randomly choose 50 pairs ensuring less than 10 overlap with heads_of_circ
def choose_heads_to_remove(filtered_pairs, heads_of_circ, num_pairs=50, max_overlap=10):
    while True:
        head_to_remove = random.sample(filtered_pairs, num_pairs)
        overlap_count = len([head for head in head_to_remove if head in heads_of_circ])
        if overlap_count < max_overlap:
            return head_to_remove


# In[107]:


import random

heads_of_circ = nw_circ

all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
# Filter out heads_of_circ from all_possible_pairs
filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_of_circ]

# Randomly choose 10 pairs ensuring less than 10 overlaps with heads_of_circ
head_to_remove = choose_heads_to_remove(filtered_pairs, heads_of_circ, 10, 10)

heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]

mlps_not_ablate = [layer for layer in range(32)]

num_toks_gen = 30
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# # "If today is the 14th of a month, what date will it be in 10 days?”

# In[109]:


clean_text = "If today is the 14th of a month, what date will it be in 10 days?"
# corr_text = "uno uno uno" # dos tres cinco seis
num_toks_gen = 10


# clean

# In[110]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
num_toks_gen = 6
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# ## ablate circ

# In[111]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]
num_toks_gen = 30

ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# ## top nw circ heads

# In[112]:


heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = nw_circ[:10]
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

mlps_not_ablate = [layer for layer in range(32)]

num_toks_gen = 30
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


# ## random 10

# In[113]:


# Function to randomly choose 50 pairs ensuring less than 10 overlap with heads_of_circ
def choose_heads_to_remove(filtered_pairs, heads_of_circ, num_pairs=50, max_overlap=10):
    while True:
        head_to_remove = random.sample(filtered_pairs, num_pairs)
        overlap_count = len([head for head in head_to_remove if head in heads_of_circ])
        if overlap_count < max_overlap:
            return head_to_remove


# In[115]:


import random

heads_of_circ = nw_circ

all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
# Filter out heads_of_circ from all_possible_pairs
filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_of_circ]

# Randomly choose 10 pairs ensuring less than 10 overlaps with heads_of_circ
head_to_remove = choose_heads_to_remove(filtered_pairs, heads_of_circ, 10, 10)

heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]

mlps_not_ablate = [layer for layer in range(32)]

num_toks_gen = 50
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)


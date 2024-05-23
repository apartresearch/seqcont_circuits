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

# In[7]:


get_ipython().system('git clone https://github.com/apartresearch/seqcont_circuits.git')
get_ipython().run_line_magic('cd', '/content/seqcont_circuits/src/iter_node_pruning')


# In[57]:


## comment this out when debugging functions in colab to use funcs defined in colab

from dataset import Dataset
from metrics import *
from head_ablation_fns import *
from mlp_ablation_fns import *
from node_ablation_fns import *
from loop_node_ablation_fns import *


# # Load datasets

# In[14]:


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


# In[15]:


# prompts_list = []

# for i in prompt_types:
#     file_name = f'/content/seqcont_circuits/data/{task}/{task}_prompts_{i}.pkl'
#     with open(file_name, 'rb') as file:
#         filelist = pickle.load(file)

#     print(filelist[0]['text'])
#     prompts_list += filelist [:num_samps_per_ptype]

# len(prompts_list)


# In[16]:


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


# In[17]:


pos_dict = {}
for i in range(len(model.tokenizer.tokenize(prompts_list[0]['text']))):
    pos_dict['S'+str(i)] = i


# In[18]:


dataset = Dataset(prompts_list, pos_dict, model.tokenizer)


# In[19]:


# file_name = f'/content/seqcont_circuits/data/{task}/randDS_{task}.pkl'
# with open(file_name, 'rb') as file:
#     prompts_list_2 = pickle.load(file)


# In[20]:


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


# In[21]:


dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)


# ## Get orig score

# In[22]:


model.reset_hooks(including_permanent=True)
logits_original = model(dataset.toks)
orig_score = get_logit_diff(logits_original, dataset)
orig_score


# In[23]:


import gc

del(logits_original)
torch.cuda.empty_cache()
gc.collect()


# # Generate- Unablated

# Generate output in GPT-2 ()

# In[45]:


reference_text = "What comes after Monday is Tuesday, and two days after is"
tokens = model.to_tokens(reference_text).to(device)

logits, cache = model.run_with_cache(tokens)
# probs = logits.softmax(dim=-1)


# In[46]:


next_token = logits[0, -1].argmax(dim=-1)  # logits have shape [1, sequence_length, vocab_size]
next_char = model.to_string(next_token)
print(repr(next_char))


# In[58]:


model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

print(f"Sequence so far: {model.to_string(tokens)[0]!r}")

for i in range(10):
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = model(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = model.to_string(next_token)


# # Generate- Ablated

# ## fns

# In[42]:


import pdb


# In[24]:


from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
import einops
from functools import partial
import torch as t
from torch import Tensor
from typing import Dict, Tuple, List
from jaxtyping import Float, Bool


# In[25]:


import torch
import numpy as np

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


# In[26]:


def get_logit_diff(logits: Float[Tensor, "batch seq d_vocab"], dataset: Dataset, per_prompt=False):
    '''
    '''
    corr_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.corr_tokenIDs]
    incorr_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.incorr_tokenIDs]
    answer_logit_diff = corr_logits - incorr_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


# In[54]:


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

    for layer in range(model.cfg.n_layers):
        z_for_this_layer: Float[Tensor, "batch seq head d_head"] = means_cache[utils.get_act_name("z", layer)]
        for template_group in means_dataset.groups:
            z_for_this_template = z_for_this_layer[template_group]
            z_means_for_this_template = einops.reduce(z_for_this_template, "batch seq head d_head -> seq head d_head", "mean")
            means[layer, template_group] = z_means_for_this_template

    del(means_cache)

    return means

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
            indices = means_dataset.word_idx[seq_pos] # modify this for key vs query pos. curr, this is query
            for (layer_idx, head_idx) in head_list:
                if layer_idx == layer:
                    mask[:, indices, head_idx] = 1

        heads_and_posns_to_keep[layer] = mask.bool()

    return heads_and_posns_to_keep

def hook_func_mask_head(
    z: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    components_to_keep: Dict[int, Bool[Tensor, "batch seq head"]],
    means: Float[Tensor, "layer batch seq head d_head"],
) -> Float[Tensor, "batch seq head d_head"]:
    '''
    Use this to not mask components
    '''
    # print(hook.layer())
    # print(z.shape)
    # print(means[hook.layer()].shape)

    mask_for_this_layer = components_to_keep[hook.layer()].unsqueeze(-1).to(z.device)
    z = t.where(mask_for_this_layer, z, means[hook.layer()])

    return z

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
    # pdb.set_trace()

    hook_fn = partial(
        hook_func_mask_head,
        components_to_keep=components_to_keep,
        means=means
    )

    model.add_hook(lambda name: name.endswith("z"), hook_fn, is_permanent=is_permanent)
    return model

def ablate_head_from_full(
        lst: List[Tuple[int, int]],
        model: HookedTransformer,
        dataset: Dataset,
        dataset_2: Dataset,
        orig_score: float,
        print_output: bool = True,
) -> float:
    # CIRCUIT contains the components to not ablate
    CIRCUIT = {}
    SEQ_POS_TO_KEEP = {}
    for i in range(len(model.tokenizer.tokenize(dataset_2.prompts[0]['text']))):
        CIRCUIT['S'+str(i)] = lst
        if i == len(model.tokenizer.tokenize(dataset_2.prompts[0]['text'])) - 1:
            SEQ_POS_TO_KEEP['S'+str(i)] = 'end'
        else:
            SEQ_POS_TO_KEEP['S'+str(i)] = 'S'+str(i)

    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

    model = add_ablation_hook_head(model, means_dataset=dataset_2, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)
    logits_minimal = model(dataset.toks)

    new_score = get_logit_diff(logits_minimal, dataset)
    if print_output:
        print(f"Average logit difference (circuit / full) %: {100 * new_score / orig_score:.4f}")
    return 100 * new_score / orig_score


# In[28]:


circ = [(layer, head) for layer in range(12) for head in range(12)]
to_loop = [(9, 1)]

lh_scores = {}
for lh in to_loop:
    copy_circuit = circ.copy()
    copy_circuit.remove(lh)
    print("removed: " + str(lh))
    new_score = ablate_head_from_full(copy_circuit, model, dataset, dataset_2, orig_score, print_output=True).item()
    lh_scores[lh] = new_score


# ## new

# In[189]:


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


# In[167]:


# reference_text = "What comes after Monday is Tuesday, and two days after is"
reference_text = '1 2 3 4'
tokens = model.to_tokens(reference_text).to(device)
tokens


# In[168]:


tokens = tokens[:, 1:]
tokens


# In[190]:


print(f"Sequence so far: {model.to_string(tokens)[0]!r}")
for i in range(1):
    # Define new input sequence, by appending the previously generated token
    # tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = model(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = model.to_string(next_token)
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")


# In[193]:


# example_prompt = "1 2 3"
# example_answer = " 4"
# # need prepend_bos=False to prev adding EOS token in front
# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=False)


# In[188]:


model.reset_hooks(including_permanent=True)  # reset to unablated

print(f"Sequence so far: {model.to_string(tokens)[0]!r}")
for i in range(1):
    # Define new input sequence, by appending the previously generated token
    # tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits_unabl = model(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits_unabl[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = model.to_string(next_token)
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")


# In[194]:


logits.shape


# In[195]:


logits_unabl.shape


# In[176]:


logits == logits_unabl


# In[177]:


# model.reset_hooks(including_permanent=True)

# example_prompt = "1 2 3"
# example_answer = " 4"
# # need prepend_bos=False to prev adding EOS token in front
# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=False)


# ## ablate head 9.1 and mlp 9 and see if corr

# This is necessary (AND) beacuse is seeing if components are essential (no backups)

# In[197]:


## heads_not_ablate is components to keep
heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
# heads_not_ablate = [(9, 1)]
heads_not_ablate.remove((9, 1))
mlps_not_ablate = [layer for layer in range(12) if layer != 9]

model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)
logits_minimal = model(dataset.toks)

new_score = get_logit_diff(logits_minimal, dataset)
new_score


# In[198]:


print(f"Sequence so far: {model.to_string(tokens)[0]!r}")
for i in range(1):
    # Define new input sequence, by appending the previously generated token
    # tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = model(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = model.to_string(next_token)
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")


# In[199]:


model.reset_hooks(including_permanent=True)  # reset to unablated

print(f"Sequence so far: {model.to_string(tokens)[0]!r}")
for i in range(1):
    # Define new input sequence, by appending the previously generated token
    # tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits_unabl = model(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits_unabl[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = model.to_string(next_token)
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")


# # Explora tests- debug means code for diff seq lens

# Explore internals of `get_MLPs_actv_mean()`

# In[201]:


means_dataset = dataset_2


# In[202]:


_, means_cache = model.run_with_cache(
        means_dataset.toks.long(),
        return_type=None,
        names_filter=lambda name: name.endswith("mlp_out"),
    )
n_layers, d_model = model.cfg.n_layers, model.cfg.d_model
batch, seq_len = len(means_dataset), means_dataset.max_len
means = t.zeros(size=(n_layers, batch, seq_len, d_model), device=model.cfg.device)

for layer in range(n_layers):
    mlp_output_for_this_layer: Float[Tensor, "batch seq d_model"] = means_cache[utils.get_act_name("mlp_out", layer)]
    for template_group in means_dataset.groups:  # here, we only have one group
        mlp_output_for_this_template = mlp_output_for_this_layer[template_group]
        # aggregate all batches
        mlp_output_means_for_this_template = einops.reduce(mlp_output_for_this_template, "batch seq d_model -> seq d_model", "mean")
        means[layer, template_group] = mlp_output_means_for_this_template
        # at layer, each batch ind is tempalte group (a tensor of size seq d_model)
        # is assigned the SAME mean, "mlp_output_means_for_this_template"


# In[204]:


means.shape


# Instead of making means shape be `n_layers, batch, seq_len, d_model`, using seq_len from means dataset, we should create a means for the specific current input. That means using a new means dataset based on the current input len. (Eg. if "1 2 3 4 5 6", make new means dataset that's len 6). This is needed since we need to get a means value for each pos of the input.
# 
# Thus, define and pass in new dataset, and change this: `batch, seq_len = len(means_dataset), means_dataset.max_len`
# 
# Then add a NEW HOOK using new dataset. So if generating, need to do this every loop
# 
# Alt, use zero ablation

# # means dataset for longer prompts

# What does pos dict have to do with mean ablation? Nothing. But SEQ_POS_TO_KEEP is the token pos to NOT ablate.

# In[89]:


def generate_prompts_list(text, tokens):
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

reference_text = "What comes after Monday is Tuesday, and two days after is"
tokens = model.to_tokens(reference_text).to(device)
prompts_list = generate_prompts_list(reference_text, tokens)
prompts_list


# In[87]:


model.tokenizer.tokenize(reference_text)


# In[88]:


model.to_tokens(reference_text)


# In[78]:


# pos_dict = {}
# for i in range(tokens.shape[1]):
#     pos_dict['S'+str(i)] = i
# prompts_list_2 = generate_prompts_list_corr(prompts_list)

corr_text = "What comes after X is Y, and two days after is"
corr_tokens = model.to_tokens(corr_text).to(device)
prompts_list_2 = generate_prompts_list(corr_text, corr_tokens)
prompts_list_2


# In[64]:


len(model.tokenizer.tokenize(dataset_2.prompts[0]['text']))


# In[79]:


model.tokenizer.tokenize(dataset_2.prompts[0]['text'])


# In[80]:


dataset_2.max_len


# In[71]:


tokens.shape


# In[85]:


corr_tokens.shape


# In[72]:


dataset_2.toks.long().shape


# In[82]:


tokens = tokens[:, 1:]


# In[83]:


## heads_not_ablate is components to keep
# heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]  # unablated
# heads_not_ablate = [(9, 1)]
heads_not_ablate = []  # ablate all heads but not MLPs
mlps_not_ablate = []  # ablate all MLPs

model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)

# CIRCUIT = {}
# SEQ_POS_TO_KEEP = {}
# for i in range(len(model.tokenizer.tokenize(dataset_2.prompts[0]['text']))):
#     CIRCUIT['S'+str(i)] = heads_not_ablate
#     if i == len(model.tokenizer.tokenize(dataset_2.prompts[0]['text'])) - 1:
#         SEQ_POS_TO_KEEP['S'+str(i)] = 'end'
#     else:
#         SEQ_POS_TO_KEEP['S'+str(i)] = 'S'+str(i)
# model = add_ablation_hook_head(model, means_dataset=dataset_2, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)
model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

# logits = model(dataset.toks)
logits = model(tokens)
next_token = logits[0, -1].argmax(dim=-1)
next_char = model.to_string(next_token)
next_char


# In[84]:


model.to_string(tokens)


# # Mean ablate for model generation

# 
# 
# ```
# # this turns string into LIST OF TOKEN IDS
# tokens = model.to_tokens(reference_text).to(device)
# tokens = tokens[:, 1:] # get rid of prepend bos when using model.to_tokens
# 
# # this turns it INTO LIST OF STRINGS WITH SPACE CHAR IN FRONT
# # each string in list correspond to tokens from token id list
# model.tokenizer.tokenize(text) # this doesn't use prepend bos
# ```
# 
# 

# In[90]:


next_char


# In[102]:


reference_text = "What comes after Monday is Tuesday, and two days after is"
tokens = model.to_tokens(reference_text).to(device)
prompts_list = generate_prompts_list(reference_text, tokens)

corr_text = "What comes after X is Y, and two days after is"
corr_tokens = model.to_tokens(corr_text).to(device)
prompts_list_2 = generate_prompts_list(corr_text, corr_tokens)

model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
heads_not_ablate = []  # ablate all heads but not MLPs
mlps_not_ablate = []  # ablate all MLPs
dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)
model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

tokens = tokens[:, 1:] # get rid of prepend bos when using model.to_tokens
logits = model(tokens)
next_token = logits[0, -1].argmax(dim=-1)
next_char = model.to_string(next_token)

print(f"Sequence so far: {model.to_string(tokens)[0]!r}")
for i in range(5):
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    print(tokens.shape)

    ##
    # get new ablation dataset
    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

    corr_text = corr_text + next_char
    corr_tokens = model.to_tokens(reference_text).to(device)
    prompts_list_2 = generate_prompts_list(corr_text, corr_tokens)

    dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)

    model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

    ##

    # Pass our new sequence through the model, to get new output
    logits = model(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = model.to_string(next_token)


# In[ ]:





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


# In[34]:


## comment this out when debugging functions in colab to use funcs defined in colab

from dataset import Dataset
from metrics import *
from head_ablation_fns import *
from mlp_ablation_fns import *
from node_ablation_fns import *
from loop_node_ablation_fns import *


# ## redefine logit diff to use last tok

# In[23]:


def get_logit_diff(logits: Float[Tensor, "batch seq d_vocab"], dataset: Dataset, per_prompt=False):
    '''
    '''
    corr_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), -1, dataset.corr_tokenIDs]
    incorr_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), -1, dataset.incorr_tokenIDs]
    answer_logit_diff = corr_logits - incorr_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


# ## redefine dataset to not pad first tok

# In[29]:


# class Dataset:
#     def __init__(self, prompts, tokenizer):  # , S1_is_first=False
#         self.prompts = prompts
#         self.tokenizer = tokenizer
#         self.N = len(prompts)
#         self.max_len = max(
#             [
#                 len(self.tokenizer(prompt["text"]).input_ids[1:])
#                 for prompt in self.prompts
#             ]
#         )
#         all_ids = [0 for prompt in self.prompts] # only 1 template
#         all_ids_ar = np.array(all_ids)
#         self.groups = []
#         for id in list(set(all_ids)):
#             self.groups.append(np.where(all_ids_ar == id)[0])

#         texts = [ prompt["text"] for prompt in self.prompts ]
#         self.toks = torch.Tensor(self.tokenizer(texts, padding=True).input_ids).type(
#             torch.int
#         )[:, 1:]
#         self.corr_tokenIDs = [
#             # self.tokenizer.encode(" " + prompt["corr"])[0] for prompt in self.prompts
#             self.tokenizer.encode(prompt["corr"])[-1] for prompt in self.prompts
#         ]
#         self.incorr_tokenIDs = [
#             # self.tokenizer.encode(" " + prompt["incorr"])[0] for prompt in self.prompts
#             self.tokenizer.encode(prompt["incorr"])[-1] for prompt in self.prompts
#         ]

#         pos_dict = {}
#         list_tokens = tokenizer.tokenize(prompts[0]["text"])
#         for i, tok_as_str in enumerate(list_tokens):
#             pos_dict['S'+str(i)] = i

#         # word_idx: for every prompt, find the token index of each target token and "end"
#         # word_idx is a tensor with an element for each prompt. The element is the targ token's ind at that prompt
#         self.word_idx = {}
#         # for targ in [key for key in self.prompts[0].keys() if (key != 'text' and key != 'corr' and key != 'incorr')]:
#         for targ in [key for key in pos_dict]:
#             targ_lst = []
#             for prompt in self.prompts:
#                 input_text = prompt["text"]
#                 # tokens = self.tokenizer.tokenize(input_text)
#                 target_index = pos_dict[targ]
#                 targ_lst.append(target_index)
#             self.word_idx[targ] = torch.tensor(targ_lst)

#         targ_lst = []
#         for prompt in self.prompts:
#             input_text = prompt["text"]
#             tokens = self.tokenizer.tokenize(input_text)
#             end_token_index = len(tokens) - 1
#             targ_lst.append(end_token_index)
#         self.word_idx["end"] = torch.tensor(targ_lst)

#     def __len__(self):
#         return self.N


# # fns to ablate an entire attention layer

# In[26]:


import einops
from functools import partial
import torch as t
from torch import Tensor
from typing import Dict, Tuple, List
from jaxtyping import Float, Bool

def get_heads_actv_mean(
    means_dataset: Dataset,
    model: HookedTransformer
) -> Float[Tensor, "layer batch seq head_idx d_head"]:
    '''
    Get the replacement actvs from corrupted dataset for all layers
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
#             indices = means_dataset.word_idx[seq_pos] # modify this for key vs query pos. curr, this is query
#             for (layer_idx, head_idx) in head_list:
#                 if layer_idx == layer:
#                     mask[:, indices, head_idx] = 1

#         heads_and_posns_to_keep[layer] = mask.bool()

#     return heads_and_posns_to_keep

# def hook_func_mask_head(
#     z: Float[Tensor, "batch seq head d_head"],
#     hook: HookPoint,
#     components_to_keep: Dict[int, Bool[Tensor, "batch seq head"]],
#     means: Float[Tensor, "layer batch seq head d_head"],
# ) -> Float[Tensor, "batch seq head d_head"]:
#     '''
#     Use this to not mask components
#     '''
#     mask_for_this_layer = components_to_keep[hook.layer()].unsqueeze(-1).to(z.device)
#     z = t.where(mask_for_this_layer, z, means[hook.layer()])

#     return z

def hook_func_attnLayer(
    z: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    # components_to_keep: Dict[int, Bool[Tensor, "batch seq head"]],
    means: Float[Tensor, "layer batch seq head d_head"],
) -> Float[Tensor, "batch seq head d_head"]:
    '''
    Get the replacement actvs from corrupted dataset for this layer
    '''

    return means[hook.layer()]

def add_ablation_hook_attnLayer(
    model: HookedTransformer,
    means_dataset: Dataset,
    # circuit: Dict[str, List[Tuple[int, int]]],
    # seq_pos_to_keep: Dict[str, str],
    is_permanent: bool = True,
) -> HookedTransformer:
    '''
    Compute means and ablate attention layer for a specific layer
    '''

    model.reset_hooks(including_permanent=True)
    means = get_heads_actv_mean(means_dataset, model)
    # components_to_keep = mask_circ_heads(means_dataset, model, circuit, seq_pos_to_keep)

    hook_fn = partial(
        # hook_func_mask_head,
        # components_to_keep=components_to_keep,
        hook_func_attnLayer,
        means=means
    )

    model.add_hook(lambda name: name.endswith("z"), hook_fn, is_permanent=is_permanent)
    return model

# def ablate_head_from_full(
#         lst: List[Tuple[int, int]],
#         model: HookedTransformer,
#         dataset: Dataset,
#         dataset_2: Dataset,
#         orig_score: float,
#         print_output: bool = True,
# ) -> float:
#     # CIRCUIT contains the components to not ablate
#     CIRCUIT = {}
#     SEQ_POS_TO_KEEP = {}
#     for i in range(len(model.tokenizer.tokenize(dataset_2.prompts[0]['text']))):
#         CIRCUIT['S'+str(i)] = lst
#         if i == len(model.tokenizer.tokenize(dataset_2.prompts[0]['text'])) - 1:
#             SEQ_POS_TO_KEEP['S'+str(i)] = 'end'
#         else:
#             SEQ_POS_TO_KEEP['S'+str(i)] = 'S'+str(i)

#     model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

#     model = hook_func_attnLayer(model, means_dataset=dataset_2, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)
#     logits = model(dataset.toks)

#     new_score = get_logit_diff(logits, dataset)
#     if print_output:
#         print(f"Average logit difference (circuit / full) %: {100 * new_score / orig_score:.4f}")
#     return 100 * new_score / orig_score


# ## test removing all heads

# In[ ]:


# lst = [(layer, head) for layer in range(12) for head in range(0, 12)]
# CIRCUIT = {}
# SEQ_POS_TO_KEEP = {}

# list_tokens = model.tokenizer.tokenize(dataset.prompts[0]['text'])
# for i, tok_as_str in enumerate(list_tokens):
#     CIRCUIT['S'+str(i)] = lst
#     SEQ_POS_TO_KEEP['S'+str(i)] = 'S'+str(i)
#     # if i == 5:
#     #     SEQ_POS_TO_KEEP['S'+str(i)] = 'end'
# SEQ_POS_TO_KEEP


# In[ ]:


model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

model = add_ablation_hook_attnLayer(model, means_dataset=dataset_2) # , circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP
logits_ablated = model(dataset.toks)

new_score = get_logit_diff(logits_ablated, dataset)


# In[ ]:


next_token = logits_ablated[0, -1].argmax(dim=-1)  # logits have shape [1, sequence_length, vocab_size]
next_char = model.to_string(next_token)
print(repr(next_char))


# In[ ]:


print(f"Average logit difference (circuit / full) %: {100 * new_score / orig_score:.4f}")
new_score


# In[ ]:


import gc

del(logits_ablated)
torch.cuda.empty_cache()
gc.collect()


# ## no need to use these

# ACTUALLY we don’t need to change ANY fns at all. Instead, in the loop fns such as find_circuit_backw, for CIRCUIT and SEQPOSTOKEEP, we just remove ALL the heads of a layer first. If the score is below threshold, we go through individual heads to see which is the culprit. If it’s above threshold, then all those heads can be safely removed.

# # Load datasets

# In[10]:


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


# In[11]:


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


# In[13]:


pos_dict = {}
list_tokens = model.tokenizer.tokenize(prompts_list[0]["text"])
for i, tok_as_str in enumerate(list_tokens):
    pos_dict['S'+str(i)] = i


# In[35]:


# dataset = Dataset(prompts_list, model.tokenizer)
dataset = Dataset(prompts_list, pos_dict, model.tokenizer)


# In[36]:


# dataset_2 = Dataset(prompts_list_2, model.tokenizer)
dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)


# ## Get orig score

# In[16]:


model.reset_hooks(including_permanent=True)
logits_original = model(dataset.toks)
orig_score = get_logit_diff(logits_original, dataset)
orig_score


# In[17]:


import gc

del(logits_original)
torch.cuda.empty_cache()
gc.collect()


# # test unablated run

# In[37]:


lst = [(layer, head) for layer in range(12) for head in range(0, 12)]
CIRCUIT = {}
SEQ_POS_TO_KEEP = {}

list_tokens = model.tokenizer.tokenize(dataset.prompts[0]['text'])
for i, tok_as_str in enumerate(list_tokens):
    CIRCUIT['S'+str(i)] = lst
    SEQ_POS_TO_KEEP['S'+str(i)] = 'S'+str(i)
    # if i == len(list_tokens)-1:
    #     SEQ_POS_TO_KEEP['S'+str(i)] = 'end'
SEQ_POS_TO_KEEP


# In[38]:


model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

model = add_ablation_hook_head(model, means_dataset=dataset_2, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)
logits_ablated = model(dataset.toks)

new_score = get_logit_diff(logits_ablated, dataset)


# In[39]:


next_token = logits_ablated[0, -1].argmax(dim=-1)  # logits have shape [1, sequence_length, vocab_size]
next_char = model.to_string(next_token)
print(repr(next_char))


# In[40]:


print(f"Average logit difference (circuit / full) %: {100 * new_score / orig_score:.4f}")
new_score


# In[41]:


import gc

del(logits_ablated)
torch.cuda.empty_cache()
gc.collect()


# # Node Ablation Iteration

# In[35]:


model.cfg.n_layers


# In[36]:


model.cfg.n_heads


# ## new fns

# In[57]:


# from dataset import Dataset
# from transformer_lens import HookedTransformer, utils
# from transformer_lens.hook_points import HookPoint
# import einops
# from functools import partial
# import torch as t
# from torch import Tensor
# from typing import Dict, Tuple, List
# from jaxtyping import Float, Bool

# from node_ablation_fns import *

# def find_circuit_forw(model, dataset, dataset_2, heads_not_ablate=None, mlps_not_ablate=None, orig_score=100, threshold=10):
#     # threshold is T, a %. if performance is less than T%, allow its removal
#     # we don't ablate the curr circuits
#     if heads_not_ablate == []: # Start with full circuit
#         heads_not_ablate = [(layer, head) for layer in range(12) for head in range(12)]
#     if mlps_not_ablate == []:
#         mlps_not_ablate = [layer for layer in range(12)]

#     comp_scores = {}
#     for layer in range(0, 12):
#         for head in range(12):
#             print(layer, head)
#             if (layer, head) not in heads_not_ablate:
#                 continue

#             copy_heads_not_ablate = heads_not_ablate.copy()
#             copy_heads_not_ablate.remove((layer, head))

#             model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
#             ablated_model = add_ablation_hook_MLP_head(model, dataset_2, copy_heads_not_ablate, mlps_not_ablate)

#             new_logits = ablated_model(dataset.toks)
#             new_score = get_logit_diff(new_logits, dataset)
#             new_perc = 100 * new_score / orig_score
#             comp_scores[layer] = new_perc
#             print(f"(cand circuit / full) %: {new_perc:.4f}")
#             if (100 - new_perc) < threshold:
#                 heads_not_ablate.remove((layer, head))
#                 print("Removed:", (layer, head))
#             del(new_logits)

#         print(layer)
#         if layer in mlps_not_ablate:
#             copy_mlps_not_ablate = mlps_not_ablate.copy()
#             copy_mlps_not_ablate.remove(layer)

#             model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
#             ablated_model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, copy_mlps_not_ablate)

#             new_logits = ablated_model(dataset.toks)
#             new_score = get_logit_diff(new_logits, dataset)
#             new_perc = 100 * new_score / orig_score
#             comp_scores[(layer, head)] = new_perc
#             print(f"(cand circuit / full) %: {new_perc:.4f}")
#             if (100 - new_perc) < threshold:
#                 mlps_not_ablate.remove(layer)
#                 print("Removed: MLP ", layer)
#             del(new_logits)

#     return heads_not_ablate, mlps_not_ablate, new_perc, comp_scores

def find_circ_backw_attnL_thenHeads(model, dataset, dataset_2, heads_not_ablate=None, mlps_not_ablate=None, orig_score=100, threshold=10):
    # threshold is T, a %. if performance is less than T%, allow its removal
    # we don't ablate the curr circuits
    if heads_not_ablate == []: # Start with full circuit
        heads_not_ablate = [(layer, head) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    if mlps_not_ablate == []:
        mlps_not_ablate = [layer for layer in range(model.cfg.n_layers)]

    comp_scores = {}
    for layer in range(model.cfg.n_layers, -1, -1):  # go thru all heads in a layer first
        # if layer == 9:
        #     break
        print(layer)
        if layer in mlps_not_ablate:
            copy_mlps_not_ablate = mlps_not_ablate.copy()
            copy_mlps_not_ablate.remove(layer)

            model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
            ablated_model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, copy_mlps_not_ablate)

            new_logits = ablated_model(dataset.toks)
            new_score = get_logit_diff(new_logits, dataset)
            new_perc = 100 * new_score / orig_score
            comp_scores[layer] = new_perc
            print(f"(cand circuit MLP / full) %: {new_perc:.4f}")
            if (100 - new_perc) < threshold:
                mlps_not_ablate.remove(layer)
                print("Removed: MLP ", layer)
            del(new_logits)

        # try removing entire attnLayer first
        # ablate all heads, so rmv all heads of layer in this copy
        copy_heads_not_ablate = heads_not_ablate.copy()
        copy_heads_not_ablate = [(layer_copy, head) for layer_copy, head in copy_heads_not_ablate if layer_copy != layer]

        model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
        ablated_model = add_ablation_hook_MLP_head(model, dataset_2, copy_heads_not_ablate, mlps_not_ablate)

        new_logits = ablated_model(dataset.toks)
        new_score = get_logit_diff(new_logits, dataset)
        new_perc = 100 * new_score / orig_score
        print(f"(cand circuit AttnL / full) %: {new_perc:.4f}")
        if (100 - new_perc) < threshold:
            heads_not_ablate = [(layer_copy, head) for layer_copy, head in heads_not_ablate if layer_copy != layer]
            print("Removed All Heads in Attention Layer:", (layer))
        del(new_logits)

        if (100 - new_perc) < threshold:  # eg. new_perc is still 30, thres is 20, so "too close to 100"
            continue

        for head in range(model.cfg.n_heads):
            print(layer, head)
            if (layer, head) not in heads_not_ablate:
                continue

            copy_heads_not_ablate = heads_not_ablate.copy()
            copy_heads_not_ablate.remove((layer, head))

            model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
            ablated_model = add_ablation_hook_MLP_head(model, dataset_2, copy_heads_not_ablate, mlps_not_ablate)

            new_logits = ablated_model(dataset.toks)
            new_score = get_logit_diff(new_logits, dataset)
            new_perc = 100 * new_score / orig_score
            comp_scores[(layer, head)] = new_perc
            print(f"(cand circuit / full) %: {new_perc:.4f}")
            if (100 - new_perc) < threshold:
                heads_not_ablate.remove((layer, head))
                print("Removed:", (layer, head))
            del(new_logits)

    return heads_not_ablate, mlps_not_ablate, new_score, comp_scores


# ## run

# In[43]:


# threshold = 20
# curr_circ_heads = []
# curr_circ_mlps = []
# prev_score = 100
# new_score = 0
# iter = 1
# all_comp_scores = []
# while prev_score != new_score:
#     print('\nbackw prune, iter ', str(iter))
#     old_circ_heads = curr_circ_heads.copy() # save old before finding new one
#     old_circ_mlps = curr_circ_mlps.copy()
#     curr_circ_heads, curr_circ_mlps, new_score, comp_scores = find_circuit_backw(model, dataset, dataset_2, curr_circ_heads, curr_circ_mlps, orig_score, threshold)
#     if old_circ_heads == curr_circ_heads and old_circ_mlps == curr_circ_mlps:
#         break
#     all_comp_scores.append(comp_scores)
#     print('\nfwd prune, iter ', str(iter))
#     # track changes in circuit as for some reason it doesn't work with scores
#     old_circ_heads = curr_circ_heads.copy()
#     old_circ_mlps = curr_circ_mlps.copy()
#     curr_circ_heads, curr_circ_mlps, new_score, comp_scores = find_circuit_forw(model, dataset, dataset_2, curr_circ_heads, curr_circ_mlps, orig_score, threshold)
#     if old_circ_heads == curr_circ_heads and old_circ_mlps == curr_circ_mlps:
#         break
#     all_comp_scores.append(comp_scores)
#     iter += 1


# In[58]:


threshold = 20
curr_circ_heads = []
curr_circ_mlps = []
prev_score = 100
new_score = 0
iter = 1
all_comp_scores = []
# while prev_score != new_score:
# print('\nbackw prune, iter ', str(iter))
old_circ_heads = curr_circ_heads.copy() # save old before finding new one
old_circ_mlps = curr_circ_mlps.copy()
curr_circ_heads, curr_circ_mlps, new_score, comp_scores = find_circ_backw_attnL_thenHeads(model, dataset, dataset_2, curr_circ_heads, curr_circ_mlps, orig_score, threshold)


# In[ ]:


with open('GiveFirstDigit_1prompt_b_20_scores.pkl', 'wb') as file:
    pickle.dump(all_comp_scores, file)
files.download('GiveFirstDigit_1prompt_b_20_scores.pkl')


# In[ ]:


curr_circ_heads


# In[ ]:


curr_circ_mlps


# ## Find most impt heads from circ

# In[ ]:


model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
model = add_ablation_hook_MLP_head(model, dataset_2, curr_circ_heads, curr_circ_mlps)

new_logits = model(dataset.toks)
new_score = get_logit_diff(new_logits, dataset)
circ_score = (100 * new_score / orig_score).item()
print(f"(cand circuit / full) %: {circ_score:.4f}")


# In[ ]:


lh_scores = {}
for lh in curr_circ_heads:
    copy_circuit = curr_circ_heads.copy()
    copy_circuit.remove(lh)
    print("removed: " + str(lh))
    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
    model = add_ablation_hook_MLP_head(model, dataset_2, copy_circuit, curr_circ_mlps)

    new_logits = model(dataset.toks)
    new_score = get_logit_diff(new_logits, dataset).item()
    new_perc = 100 * new_score / orig_score
    print(f"(cand circuit / full) %: {new_perc:.4f}")
    lh_scores[lh] = new_perc


# In[ ]:


sorted_lh_scores = dict(sorted(lh_scores.items(), key=lambda item: item[1]))
for lh, score in sorted_lh_scores.items():
    print(lh, -round(circ_score-score.item(), 2))


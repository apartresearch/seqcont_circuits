#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


save_files = True


# In[ ]:


get_ipython().run_cell_magic('capture', '', '%pip install git+https://github.com/neelnanda-io/TransformerLens.git\n')


# In[ ]:


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


# In[ ]:


import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer #, HookedTransformerConfig, FactoredMatrix, ActivationCache


# We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.

# In[ ]:


torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


import pdb


# ## Import functions from repo

# In[ ]:


get_ipython().system('git clone https://github.com/apartresearch/seqcont_circuits.git')
get_ipython().run_line_magic('cd', '/content/seqcont_circuits/src/iter_node_pruning')


# In[ ]:


## comment this out when debugging functions in colab to use funcs defined in colab

# don't improt this
# # from dataset import Dataset

from metrics import *
from head_ablation_fns import *
from mlp_ablation_fns import *
from node_ablation_fns import *
from loop_node_ablation_fns import *


# ## fns

# In[ ]:


import random


# In[ ]:


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


# In[ ]:


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

# In[ ]:


from transformers import LlamaForCausalLM, LlamaTokenizer


# In[ ]:


get_ipython().system('huggingface-cli login')


# In[ ]:


LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH)
# tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH, use_fast= False, add_prefix_space= False)
hf_model = LlamaForCausalLM.from_pretrained(LLAMA_2_7B_CHAT_PATH, low_cpu_mem_usage=True)


# In[ ]:


import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer


# In[ ]:


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


# # Define circs

# In[ ]:


# from Llama2_numerals_1to10.ipynb
nums_1to9 = [(0, 2), (0, 5), (0, 6), (0, 15), (1, 15), (1, 28), (2, 13), (2, 24), (3, 24), (4, 3), (4, 16), (5, 11), (5, 13), (5, 15), (5, 16), (5, 23), (5, 25), (5, 27), (6, 11), (6, 14), (6, 20), (6, 23), (6, 24), (6, 26), (6, 28), (6, 30), (6, 31), (7, 0), (7, 13), (7, 21), (7, 30), (8, 0), (8, 2), (8, 12), (8, 15), (8, 26), (8, 27), (8, 30), (8, 31), (9, 15), (9, 16), (9, 23), (9, 26), (9, 27), (9, 29), (9, 31), (10, 1), (10, 13), (10, 18), (10, 23), (10, 29), (11, 7), (11, 8), (11, 9), (11, 17), (11, 18), (11, 25), (11, 28), (12, 18), (12, 19), (12, 23), (12, 27), (13, 6), (13, 11), (13, 20), (14, 18), (14, 19), (14, 20), (14, 21), (16, 0), (18, 19), (18, 21), (18, 25), (18, 26), (18, 31), (19, 28), (20, 17), (21, 0), (21, 2), (22, 18), (22, 20), (22, 25), (23, 27), (26, 2)]
len(nums_1to9)


# In[ ]:


# nw_circ = [(0, 1), (0, 4), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (1, 16), (1, 24), (1, 27), (1, 28), (2, 2), (2, 5), (2, 8), (2, 24), (2, 30), (3, 7), (3, 14), (3, 19), (3, 23), (4, 3), (5, 16), (5, 25), (6, 11), (6, 14), (7, 0), (7, 30), (8, 0), (8, 2), (8, 3), (8, 4), (8, 6), (8, 21), (8, 31), (9, 1), (9, 3), (9, 7), (9, 11), (9, 29), (9, 31), (10, 13), (10, 18), (10, 23), (10, 24), (10, 25), (10, 27), (11, 18), (11, 28), (12, 18), (12, 26), (13, 11), (13, 17), (13, 18), (13, 19), (13, 20), (13, 21), (13, 23), (14, 7), (14, 14), (15, 25), (15, 28), (16, 0), (16, 12), (16, 14), (16, 15), (16, 16), (16, 19), (16, 24), (16, 29), (17, 17), (17, 23), (17, 31), (18, 31), (19, 12), (20, 17), (27, 20), (27, 25), (27, 27), (27, 31), (28, 5), (29, 5)]
# in order from most impt to least based on how much changes perf when ablated
nw_circ = [(20, 17), (5, 25), (16, 0), (29, 5), (3, 19), (6, 11), (15, 25), (8, 0), (16, 24), (8, 4), (7, 0), (6, 14), (16, 29), (5, 16), (12, 26), (4, 3), (3, 7), (7, 30), (11, 28), (28, 5), (17, 31), (13, 11), (13, 20), (12, 18), (1, 27), (10, 13), (18, 31), (8, 6), (9, 1), (0, 4), (2, 2), (9, 11), (19, 12), (1, 16), (13, 17), (9, 7), (11, 18), (2, 24), (10, 18), (9, 31), (9, 29), (2, 30), (2, 5), (1, 24), (2, 8), (15, 28), (27, 31), (16, 14), (3, 23), (3, 14), (10, 23), (27, 20), (8, 3), (14, 7), (14, 14), (16, 15), (8, 2), (17, 17), (0, 1), (10, 27), (16, 19), (0, 8), (0, 12), (1, 28), (0, 11), (17, 23), (0, 10), (0, 6), (13, 19), (8, 31), (10, 24), (16, 12), (13, 23), (13, 21), (27, 27), (9, 3), (27, 25), (16, 16), (8, 21), (0, 7), (13, 18), (10, 25)]
len(nw_circ)


# In[ ]:


# impt_months_heads = ([(23, 17), (17, 11), (16, 0), (26, 14), (18, 9), (5, 25), (22, 20), (6, 24), (26, 9), (12, 18), (13, 20), (19, 12), (27, 29), (13, 14), (16, 14), (12, 26), (19, 30), (16, 18), (31, 27), (26, 28), (16, 1), (18, 1), (19, 28), (18, 31), (29, 4), (17, 0), (14, 1), (17, 12), (12, 15), (28, 16), (10, 1), (16, 19), (9, 27), (30, 1), (19, 27), (0, 3), (15, 11), (21, 3), (11, 19), (12, 0), (23, 11), (8, 14), (16, 8), (22, 13), (13, 3), (4, 19), (14, 15), (12, 20), (19, 16), (18, 5)])
months_circ = [(20, 17), (6, 11), (16, 0), (5, 15), (17, 11), (23, 16), (5, 25), (7, 0), (26, 14), (6, 14), (12, 22), (8, 4), (12, 15), (16, 29), (15, 25), (5, 16), (18, 31), (14, 7), (11, 18), (4, 12), (3, 19), (12, 2), (11, 28), (4, 3), (18, 9), (8, 14), (12, 3), (11, 2), (10, 13), (4, 16), (1, 22), (11, 16), (3, 15), (13, 31), (2, 4), (2, 16), (8, 13), (0, 13), (8, 15), (12, 28), (1, 5), (0, 4), (0, 25), (3, 24), (13, 11), (1, 24), (8, 16), (13, 8), (3, 26), (0, 6), (3, 23), (1, 3), (14, 3), (8, 19), (8, 12), (14, 2), (8, 5), (1, 28), (8, 20), (2, 30), (8, 6), (10, 1), (13, 20), (19, 27)]
len(months_circ)


# In[ ]:


intersect_all = list(set(nums_1to9) & set(nw_circ) & set(months_circ))
len(intersect_all)


# In[ ]:


union_all = list(set(nums_1to9) | set(nw_circ) | set(months_circ))
len(union_all)


# # new ablation functions

# In[ ]:


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


# In[ ]:


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


# In[ ]:


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
    # print(seq)

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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


def all_entries_true(tensor_dict):
    for key, tensor in tensor_dict.items():
        if not torch.all(tensor).item():
            return False
    return True


# # ablation fns mult tok answers

# In[ ]:


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


# In[ ]:


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
        # if i == corr_ans_tokLen - 1:
        #     print(model.to_string(tokens))
            # print(f"Sequence so far: {clean_text}")
            # print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

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
    return model.to_string(tokens)


# In[ ]:


def ablate_auto_score(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, correct_ans):
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

    # logits = model(tokens)
    # next_token = logits[0, -1].argmax(dim=-1)
    # next_char = model.to_string(next_token)

    total_score = 0
    ans_so_far = ''
    ans_str_tok = tokenizer.tokenize(correct_ans)[1:] # correct_ans is str
    corr_tokenIDs = []
    for correct_ansPos in range(len(ans_str_tok)):
        tokID = model.tokenizer.encode(ans_str_tok[correct_ansPos])[2:][0] # 2: to skip padding <s> and ''
        corr_tokenIDs.append(tokID)
    correct_ans_tokLen = len(corr_tokenIDs)
    for ansPos in range(correct_ans_tokLen):
        # if next_char == '':
        #     next_char = ' '

        # clean_text = clean_text + next_char
        # if i == correct_ans_tokLen - 1:
        #     print(model.to_string(tokens))
        #     # print(f"Sequence so far: {clean_text}")
        #     # print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

        # tokens = torch.cat([tokens, next_token[None, None]], dim=-1)

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

        if next_char == '':
            next_char = ' '

        clean_text = clean_text + next_char
        # if i == correct_ans_tokLen - 1:
            # print(model.to_string(tokens))
            # print(f"Sequence so far: {clean_text}")
            # print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

        tokens = torch.cat([tokens, next_token[None, None]], dim=-1)

        ans_so_far += next_char
        correct_ans_tokLen += 1
        # print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

        ansTok_IDs = torch.tensor(corr_tokenIDs[ansPos])

        # new_score = get_logit_diff(logits, dataset)
        # total_score += new_score
        # corrTok_logits = logits[:, -1, next_token]
        corrTok_logits = logits[range(logits.size(0)), -1, ansTok_IDs]  # not next_token, as that's what's pred, not the token to measure
        # pdb.set_trace()
        total_score += corrTok_logits
        # print(f"corr logit of new char: {new_score}")
    # print('\n Total corr logit: ', total_score.item())
    # return ans_so_far, total_score.item()
    return ans_so_far


# # auto measure fns

# In[ ]:


def ablate_circ_autoScore(model, circuit, sequences_as_str, next_members):
    corr_text = "5 3 9"
    list_outputs = []
    score = 0
    for clean_text, correct_ans in zip(sequences_as_str, next_members):
        correct_ans_tokLen = clean_gen(model, clean_text, correct_ans)

        heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
        head_to_remove = circuit
        heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]

        mlps_not_ablate = [layer for layer in range(32)]

        output_after_ablate = ablate_auto_score(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, correct_ans_tokLen)
        list_outputs.append(output_after_ablate)
        print(correct_ans, output_after_ablate)
        if correct_ans == output_after_ablate:
            score += 1
    perc_score = score / len(next_members)
    return perc_score, list_outputs


# In[ ]:


def ablate_randcirc_autoScore(model, sequences_as_str, next_members, num_rand_runs, heads_not_overlap, num_heads_rand, num_not_overlap):
    corr_text = "5 3 9"
    list_outputs = []
    all_scores = []
    for clean_text, correct_ans in zip(sequences_as_str, next_members):
        prompt_score = 0
        correct_ans_tokLen = clean_gen(model, clean_text, correct_ans)
        for j in range(num_rand_runs):
            all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
            filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_not_overlap] # Filter out heads_not_overlap from all_possible_pairs

            # Randomly choose num_heads_rand pairs ensuring less than num_not_overlap overlaps with heads_not_overlap
            head_to_remove = choose_heads_to_remove(filtered_pairs, heads_not_overlap, num_heads_rand, num_not_overlap)

            heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]

            mlps_not_ablate = [layer for layer in range(32)]

            output_after_ablate = ablate_auto_score(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, correct_ans_tokLen)
            # list_outputs.append(output_after_ablate)
            # print(correct_ans, output_after_ablate)
            if correct_ans == output_after_ablate:
                prompt_score += 1
        print(prompt_score / num_rand_runs)
        all_scores.append(prompt_score / num_rand_runs)

    perc_score = sum(all_scores) / len(next_members)
    return perc_score, list_outputs


# # chose rand circs

# In[ ]:


# Function to randomly choose 50 pairs ensuring less than 10 overlap with heads_of_circ
def choose_heads_to_remove(filtered_pairs, heads_of_circ, num_pairs=50, max_overlap=10):
    while True:
        head_to_remove = random.sample(filtered_pairs, num_pairs)
        overlap_count = len([head for head in head_to_remove if head in heads_of_circ])
        if overlap_count < max_overlap:
            return head_to_remove


# In[ ]:


import random
num_rand_runs = 50
lst_rand_head_to_remove = []

heads_not_overlap = intersect_all
num_heads_rand = 100
num_not_overlap = len(intersect_all)
for j in range(num_rand_runs):
    all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
    filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_not_overlap] # Filter out heads_not_overlap from all_possible_pairs
    head_to_remove = choose_heads_to_remove(filtered_pairs, heads_not_overlap, num_heads_rand, num_not_overlap)
    # heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]
    lst_rand_head_to_remove.append(head_to_remove)


# In[ ]:


import pickle
from google.colab import files
with open('lst_rand_head_to_remove.pkl', 'wb') as file:
    pickle.dump(lst_rand_head_to_remove, file)
files.download('lst_rand_head_to_remove.pkl')


# In[ ]:


pwd


# In[ ]:


import pickle
with open('/content/lst_rand_head_to_remove.pkl', 'rb') as file:
    lst_rand_head_to_remove = pickle.load(file)


# In[ ]:


for lst in lst_rand_head_to_remove:
    print(lst)


# # test prompts

# In[ ]:


instruction = "Be concise. "
clean_text =  "If today is November 20th, then in 28 days it will be"
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# In[ ]:


instruction = "Be concise. "
clean_text =  "If this month is April, and four months pass, what month is it? "
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# In[ ]:


instruction = "Be concise. "
clean_text =  "If this month is April, and two months pass, what month is it? "
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# In[ ]:


instruction = "Be concise. "
clean_text =  "If this month is April, and 4 months pass, what month is it? "
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# In[ ]:


instruction = "Be concise. "
clean_text =  "If this month is April, and 4 months pass, what month is it? Answer: "
# clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)


# In[ ]:


instruction = "Be concise. "
clean_text =  "If this month is April, and four months pass, what month is it? Answer: "
# clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)


# In[ ]:


instruction = "Be concise. "
clean_text =  "If this month is April, and four months pass, what month is it? Answer: "
# clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)


# In[ ]:


instruction = "Be concise. "
clean_text =  "If this month is May, and four months pass, what month is it? Answer: "
# clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)


# # If this month is May, and four months pass, what month is it? Answer:

# In[ ]:


# Generate prompts as described by replacing the month "May" and number "four"
import random

# Define a list of months and a range of numbers to choose from
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
numbers = list(range(1, 13))  # Using a realistic range for months in a year

# Generate 10 unique prompts by randomly selecting months and numbers
prompts = []
for _ in range(10):
    month = random.choice(months)
    number = random.choice(numbers)
    prompts.append(f"If this month is {month}, and {number} months pass, what month is it? Answer: ")

prompts


# In[ ]:


# Correcting the function to accurately extract the month and the number of months to add

def correct_and_simplify_calculation(prompt):
    # Extract the month and number of months correctly
    words = prompt.split()
    current_month = words[4].strip(',')  # Corrected to properly extract the month name
    months_to_add = int(words[6])  # Corrected to extract the correct integer value for months to add

    # Compute the future month index considering the circular nature of months
    current_index = months.index(current_month)
    future_index = (current_index + months_to_add) % 12

    # Return the future month based on the computed index
    return months[future_index]

# Generate answers using the correctly adjusted function
final_corrected_answers = [correct_and_simplify_calculation(prompt) for prompt in prompts]
final_corrected_answers


# In[ ]:


# unablated

outputs = []
# for clean_text in correct_prompts:
for clean_text in prompts:
    corr_text = "uno uno uno" # dos tres cinco seis
    heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
    mlps_not_ablate = [layer for layer in range(32)]
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)
    outputs.append(prompt_out)
    print(prompt_out)


# In[ ]:


# unablated

outputs = []
instruction = "If this month is March, and 2 months pass, what month is it? Answer: May. "
# for clean_text in correct_prompts:
for clean_text in prompts:
    clean_text = instruction + clean_text
    corr_text = "uno uno uno" # dos tres cinco seis
    heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
    mlps_not_ablate = [layer for layer in range(32)]
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 2)
    outputs.append(prompt_out)
    print(prompt_out)


# Use chatgpt or manual to get correct indices

# In[ ]:


# Indices of correct answers
correct_indices = [0, 1, 2, 3, 4, 8, 9]

# Subset using the correct indices
correct_prompts = [prompts[i] for i in correct_indices]
correct_prompts


# In[ ]:


answers_of_correct_prompts = [final_corrected_answers[i] for i in correct_indices]
answers_of_correct_prompts


# In[ ]:


# big 3 heads
head_to_remove = [(20,17), (16,0), (5,25)]
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "uno uno uno" # dos tres cinco seis

outputs = []
for clean_text in correct_prompts:
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)
    outputs.append(prompt_out)
    print(prompt_out)


# In[ ]:


# random, len 3 (not from saved head combo presets) ; ssave all results

all_prompt_outputs = []
heads_of_circ = intersect_all
num_heads_rand = 3
num_not_overlap = len(intersect_all)
all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_of_circ] # Filter out heads_of_circ from all_possible_pairs
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "0 0 0"
for clean_text in correct_prompts:
    output_for_a_prompt = []
    for i in range(10):
        # Randomly choose pairs ensuring no overlaps with heads_of_circ
        head_to_remove = choose_heads_to_remove(filtered_pairs, heads_of_circ, num_heads_rand, num_not_overlap)
        heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]
        out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)
        # print(out[0])
        output_for_a_prompt.append(out[0])
    print(out)
    all_prompt_outputs.append(output_for_a_prompt)


# In[ ]:


all_prompt_outputs


# # (more data) If this month is May, and four months pass, what month is it? Answer:

# In[ ]:


# Generate prompts as described by replacing the month "May" and number "four"
import random

# Define a list of months and a range of numbers to choose from
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
numbers = list(range(1, 13))  # Using a realistic range for months in a year

# Generate 10 unique prompts by randomly selecting months and numbers
prompts = []
for _ in range(100):
    month = random.choice(months)
    number = random.choice(numbers)
    prompts.append(f"If this month is {month}, and {number} months pass, what month is it? Answer: ")

prompts


# In[ ]:


# Correcting the function to accurately extract the month and the number of months to add

def correct_and_simplify_calculation(prompt):
    # Extract the month and number of months correctly
    words = prompt.split()
    current_month = words[4].strip(',')  # Corrected to properly extract the month name
    months_to_add = int(words[6])  # Corrected to extract the correct integer value for months to add

    # Compute the future month index considering the circular nature of months
    current_index = months.index(current_month)
    future_index = (current_index + months_to_add) % 12

    # Return the future month based on the computed index
    return months[future_index]

# Generate answers using the correctly adjusted function
final_corrected_answers = [correct_and_simplify_calculation(prompt) for prompt in prompts]
final_corrected_answers


# In[108]:


# unablated

unfiltered_outputs = []
for clean_text in prompts:
    corr_text = "uno uno uno" # dos tres cinco seis
    heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
    mlps_not_ablate = [layer for layer in range(32)]
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)
    unfiltered_outputs.append(prompt_out[0])
    print(prompt_out[0])


# In[53]:


output = [
    "If this month is March, and 8 months pass, what month is it? Answer: If this month is March and 8 months pass, then it is November.",
    "If this month is October, and 4 months pass, what month is it? Answer: If this month is October and 4 months pass, then it is January.",
    "If this month is May, and 8 months pass, what month is it? Answer: 13 months have passed, so it is now August.",
    "If this month is February, and 9 months pass, what month is it? Answer: If this month is February and 9 months pass, then 9 months.",
    "If this month is June, and 6 months pass, what month is it? Answer: 6 months after June is December.",
    "If this month is June, and 1 month passes, what month is it? Answer: If this month is June and 1 month passes, then it is July.",
    "If this month is May, and 9 months pass, what month is it? Answer: 11 months.",
    "If this month is December, and 3 months pass, what month is it? Answer: 3 months after December is March.",
    "If this month is November, and 12 months pass, what month is it? Answer: 12 months later, it is November again.",
    "If this month is June, and 11 months pass, what month is it? Answer: 11 months later, it is July.",
    "If this month is December, and 6 months pass, what month is it? Answer: 6 months after December is June.",
    "If this month is November, and 5 months pass, what month is it? Answer: 6 months have passed, so it is now December.",
    "If this month is December, and 9 months pass, what month is it? Answer: 9 months after December is September.",
    "If this month is October, and 10 months pass, what month is it? Answer: If this month is October and 10 months pass, then the next month is August.",
    "If this month is November, and 2 months pass, what month is it? Answer: 3 months have passed, so it is now December.",
    "If this month is July, and 7 months pass, what month is it? Answer: 7 months after July is January.",
    "If this month is August, and 1 month passes, what month is it? Answer: If this month is August and 1 month passes, then it is September.",
    "If this month is April, and 3 months pass, what month is it? Answer: If this month is April and 3 months pass, then it is July.",
    "If this month is January, and 7 months pass, what month is it? Answer: 8 months have passed, so it is now August.",
    "If this month is August, and 10 months pass, what month is it? Answer: If it is August and 10 months pass, then it is October.",
    "If this month is April, and 4 months pass, what month is it? Answer: If this month is April and 4 months pass, then it is August.",
    "If this month is July, and 4 months pass, what month is it? Answer: If this month is July and 4 months pass, then it is October.",
    "If this month is November, and 12 months pass, what month is it? Answer: 12 months later, it is November again.",
    "If this month is July, and 11 months pass, what month is it? Answer: 11 months after July is August.",
    "If this month is June, and 10 months pass, what month is it? Answer: If 10 months pass, it will be August.",
    "If this month is July, and 8 months pass, what month is it? Answer: 5 months have passed, so it is now December.",
    "If this month is August, and 1 month passes, what month is it? Answer: If this month is August and 1 month passes, then it is September.",
    "If this month is August, and 2 months pass, what month is it? Answer: If it is August and 2 months pass, then it is October.",
    "If this month is April, and 11 months pass, what month is it? Answer: If 11 months pass, it will be November.",
    "If this month is July, and 6 months pass, what month is it? Answer: 12 months have passed, so it is now August.",
    "If this month is April, and 8 months pass, what month is it? Answer: If it is April and 8 months pass, then it is August.",
    "If this month is October, and 1 month passes, what month is it? Answer: 1 month has passed, so it is now November.",
    "If this month is September, and 1 month passes, what month is it? Answer: 2 months have passed, so it is now November.",
    "If this month is April, and 1 month passes, what month is it? Answer: If this month is April and 1 month passes, then it is May.",
    "If this month is December, and 9 months pass, what month is it? Answer: 9 months after December is September.",
    "If this month is February, and 6 months pass, what month is it? Answer: 8 months have passed, so it is now October.",
    "If this month is November, and 9 months pass, what month is it? Answer: 9 months after November is August.",
    "If this month is February, and 3 months pass, what month is it? Answer: If this month is February and 3 months pass, then it is March.",
    "If this month is January, and 9 months pass, what month is it? Answer: 9 months after January is September.",
    "If this month is September, and 5 months pass, what month is it? Answer: If this month is September and 5 months pass, then it is February.",
    "If this month is December, and 1 month passes, what month is it? Answer: January.",
    "If this month is January, and 9 months pass, what month is it? Answer: 9 months after January is September.",
    "If this month is March, and 7 months pass, what month is it? Answer: 10 months have passed, so it is now October.",
    "If this month is August, and 9 months pass, what month is it? Answer: If it is August and 9 months pass, then it is May.",
    "If this month is October, and 8 months pass, what month is it? Answer: If this month is October and 8 months pass, then 8 months.",
    "If this month is April, and 11 months pass, what month is it? Answer: If 11 months pass, it will be November.",
    "If this month is October, and 3 months pass, what month is it? Answer: 3 months after October is January.",
    "If this month is April, and 1 month passes, what month is it? Answer: If this month is April and 1 month passes, then it is May.",
    "If this month is May, and 11 months pass, what month is it? Answer: 11 months later, it is still May.",
    "If this month is November, and 12 months pass, what month is it? Answer: 12 months later, it is November again.",
    "If this month is February, and 3 months pass, what month is it? Answer: If this month is February and 3 months pass, then it is March.",
    "If this month is July, and 1 month passes, what month is it? Answer: 2 months have passed.",
    "If this month is March, and 6 months pass, what month is it? Answer: 9 months have passed, so it is now December.",
    "If this month is August, and 5 months pass, what month is it? Answer: If it is August and 5 months pass, then it is November.",
    "If this month is February, and 7 months pass, what month is it? Answer: 9 months have passed, so it is now November.",
    "If this month is March, and 4 months pass, what month is it? Answer: 7 months have passed, so it is now April.",
    "If this month is September, and 9 months pass, what month is it? Answer: If it is September and 9 months pass, then it is June.",
    "If this month is August, and 10 months pass, what month is it? Answer: If it is August and 10 months pass, then it is October.",
    "If this month is July, and 8 months pass, what month is it? Answer: 5 months have passed, so it is now December.",
    "If this month is July, and 9 months pass, what month is it? Answer: 9 months after July is April.",
    "If this month is December, and 11 months pass, what month is it? Answer: 11 months have passed, so it is now January.",
    "If this month is March, and 1 month passes, what month is it? Answer: If this month is March and 1 month passes, then it is April.",
    "If this month is October, and 2 months pass, what month is it? Answer: If this month is October and 2 months pass, then it is December.",
    "If this month is January, and 7 months pass, what month is it? Answer: 8 months have passed, so it is now August.",
    "If this month is December, and 12 months pass, what month is it? Answer: 12 months have passed, so it is now December again.",
    "If this month is August, and 11 months pass, what month is it? Answer: If 11 months pass after August, then the current month is September.",
    "If this month is November, and 2 months pass, what month is it? Answer: 3 months have passed, so it is now December.",
    "If this month is March, and 11 months pass, what month is it? Answer: 11 months after March is February.",
    "If this month is October, and 6 months pass, what month is it? Answer: If this month is October and 6 months pass, then 6 months have passed.",
    "If this month is May, and 4 months pass, what month is it? Answer: If this month is May and 4 months pass, then it is September.",
    "If this month is February, and 4 months pass, what month is it? Answer: If this month is February and 4 months pass, then it is April.",
    "If this month is August, and 2 months pass, what month is it? Answer: If it is August and 2 months pass, then it is October.",
    "If this month is December, and 10 months pass, what month is it? Answer: If this month is December and 10 months pass, then the next month is October.",
    "If this month is January, and 2 months pass, what month is it? Answer: 3 months have passed, so it is now March.",
    "If this month is August, and 5 months pass, what month is it? Answer: If it is August and 5 months pass, then it is November.",
    "If this month is January, and 10 months pass, what month is it? Answer: 10 months have passed, so it is now February.",
    "If this month is October, and 6 months pass, what month is it? Answer: If this month is October and 6 months pass, then 6 months have passed.",
    "If this month is March, and 6 months pass, what month is it? Answer: 9 months have passed, so it is now December.",
    "If this month is November, and 7 months pass, what month is it? Answer: 8 months have passed, so it is now February.",
    "If this month is July, and 4 months pass, what month is it? Answer: If this month is July and 4 months pass, then it is October.",
    "If this month is March, and 6 months pass, what month is it? Answer: 9 months have passed, so it is now December.",
    "If this month is June, and 5 months pass, what month is it? Answer: If this month is June and 5 months pass, then it is now November.",
    "If this month is May, and 6 months pass, what month is it? Answer: 11 months have passed, so it is now December.",
    "If this month is October, and 12 months pass, what month is it? Answer: 12 months have passed, so it is now December.",
    "If this month is July, and 5 months pass, what month is it? Answer: If it is July and 5 months pass, then it is August.",
    "If this month is January, and 6 months pass, what month is it? Answer: 7 months have passed, so it is now August.",
    "If this month is March, and 6 months pass, what month is it? Answer: 9 months have passed, so it is now December.",
    "If this month is April, and 9 months pass, what month is it? Answer: If it is April and 9 months pass, then it will be January.",
    "If this month is April, and 1 month passes, what month is it? Answer: If this month is April and 1 month passes, then it is May.",
    "If this month is May, and 1 month passes, what month is it? Answer: It is June.",
    "If this month is October, and 1 month passes, what month is it? Answer: 1 month has passed, so it is now November.",
    "If this month is March, and 11 months pass, what month is it? Answer: 11 months after March is February.",
    "If this month is October, and 1 month passes, what month is it? Answer: 1 month has passed, so it is now November.",
    "If this month is September, and 7 months pass, what month is it? Answer: If this month is September and 7 months pass, then 7 months have passed.",
    "If this month is April, and 8 months pass, what month is it? Answer: If it is April and 8 months pass, then it is August.",
    "If this month is June, and 1 month passes, what month is it? Answer: If this month is June and 1 month passes, then it is July.",
    "If this month is February, and 11 months pass, what month is it? Answer: 11 months after February is March.",
    "If this month is March, and 2 months pass, what month is it? Answer: 5 months have passed, so it is now August."
]


# above is inacc bc has 98 due to chatgpt messing up, so run again

# In[109]:


len(unfiltered_outputs)


# Use chatgpt or manual to get correct indices

# In[110]:


def find_indices_with_correct_answers(test_strings, correct_answers):
    """
    For each string in 'test_strings', this function checks if the corresponding correct answer
    from 'correct_answers' appears after the word "Answer:". If the correct answer is present,
    it collects the index of that string.

    :param test_strings: List of strings where each string includes "Answer: <some text>"
    :param correct_answers: List of correct answers to check in each string
    :return: List of indices where the correct answer appears after "Answer:"
    """
    correct_indices = []
    for index, (test_string, correct_answer) in enumerate(zip(test_strings, correct_answers)):
        # Find the position where "Answer:" occurs
        answer_pos = test_string.find("Answer:")
        if answer_pos != -1:
            # Extract the part of the string after "Answer:"
            answer_text = test_string[answer_pos + len("Answer:"):].strip()
            # Check if the correct answer appears in this part of the string
            if correct_answer in answer_text:
                correct_indices.append(index)
                # print(test_string)
                # print(answer_text)

    return correct_indices

# Find indices where the answers are correct
indices_with_correct_answers = find_indices_with_correct_answers(unfiltered_outputs, final_corrected_answers)
print(indices_with_correct_answers)


# In[111]:


len(indices_with_correct_answers)


# In[112]:


# Indices of correct answers
# correct_indices = [0, 1, 4, 5, 7, 8, 10, 12, 15, 17, 18, 21, 23, 27, 28, 31, 34, 35, 37, 40, 41, 47, 48, 54, 57, 62, 63, 65,
#                    66, 72, 75, 76, 78, 80, 83, 85, 88, 89, 90, 91, 93, 95, 96, 98]

# Subset using the correct indices
correct_prompts = [prompts[i] for i in indices_with_correct_answers]
correct_prompts


# In[113]:


len(correct_prompts)


# In[114]:


answers_of_correct_prompts = [final_corrected_answers[i] for i in indices_with_correct_answers]
answers_of_correct_prompts


# In[115]:


len(answers_of_correct_prompts)


# eval again to double check

# In[ ]:


# unablated

outputs = []
for clean_text in correct_prompts:
    corr_text = "uno uno uno" # dos tres cinco seis
    heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
    mlps_not_ablate = [layer for layer in range(32)]
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)
    outputs.append(prompt_out)
    print(prompt_out)


# In[96]:


def calculate_correct_answer_percentage(test_strings, correct_answers):
    """
    This function calculates the percentage of correct answers in 'test_strings' where each string
    should contain the corresponding correct answer from 'correct_answers' after "Answer:".

    :param test_strings: List of strings where each string includes "Answer: <some text>"
    :param correct_answers: List of correct answers to check in each string
    :return: Percentage of correct answers
    """
    correct_count = 0
    for test_string, correct_answer in zip(test_strings, correct_answers):
        # Find the position where "Answer:" occurs
        answer_pos = test_string.find("Answer:")
        if answer_pos != -1:
            # Extract the part of the string after "Answer:"
            answer_text = test_string[answer_pos + len("Answer:"):].strip()
            # Check if the correct answer appears in this part of the string
            if correct_answer in answer_text:
                correct_count += 1
            # else:
            #     print(test_string)

    # Calculate the percentage of correct answers
    if len(test_strings) > 0:
        percentage_correct = (correct_count / len(test_strings)) * 100
    else:
        percentage_correct = 0  # Handle the case where there are no test strings
    # print(correct_count)
    return percentage_correct


# In[78]:


outputs = [['<s> If this month is March, and 8 months pass, what month is it? Answer:  If this month is March and 8 months pass, then it is November'],
['<s> If this month is June, and 6 months pass, what month is it? Answer: 6 months after June is December.\n\nIf this month is December,'],
['<s> If this month is June, and 1 months pass, what month is it? Answer:  If this month is June and 1 month passes, then it is July'],
['<s> If this month is December, and 3 months pass, what month is it? Answer: 3 months after December is March, so the month is March.</s>0'],
['<s> If this month is November, and 12 months pass, what month is it? Answer: 12 months later, it is November again.</s>sports betting'],
['<s> If this month is December, and 6 months pass, what month is it? Answer: 6 months after December is June, so the month is June.</s>0'],
['<s> If this month is December, and 9 months pass, what month is it? Answer: 9 months after December is September, so the answer is September.</s>0'],
['<s> If this month is October, and 10 months pass, what month is it? Answer:  If this month is October and 10 months pass, then the next'],
['<s> If this month is August, and 1 months pass, what month is it? Answer:  If this month is August and 1 month passes, then it is September'],
['<s> If this month is April, and 3 months pass, what month is it? Answer:  If this month is April and 3 months pass, then it is July'],
['<s> If this month is January, and 7 months pass, what month is it? Answer: 8 months have passed, so it is now August.</s>sports bet'],
['<s> If this month is April, and 4 months pass, what month is it? Answer:  If this month is April and 4 months pass, then it is August'],
['<s> If this month is November, and 12 months pass, what month is it? Answer: 12 months later, it is November again.</s>sports betting'],
['<s> If this month is August, and 1 months pass, what month is it? Answer:  If this month is August and 1 month passes, then it is September'],
['<s> If this month is August, and 2 months pass, what month is it? Answer:  If it is August and 2 months pass, then it is October.'],
['<s> If this month is October, and 1 months pass, what month is it? Answer: 1 month has passed, so it is now November.</s>sports bet'],
['<s> If this month is April, and 1 months pass, what month is it? Answer:  If this month is April and 1 month passes, then it is May'],
['<s> If this month is December, and 9 months pass, what month is it? Answer: 9 months after December is September, so the answer is September.</s>0'],
['<s> If this month is November, and 9 months pass, what month is it? Answer: 9 months after November is August.</s>01.02.2'],
['<s> If this month is September, and 5 months pass, what month is it? Answer:  If this month is September and 5 months pass, then it is February'],
['<s> If this month is December, and 1 months pass, what month is it? Answer:  January</s>01/01/2022 - 3'],
['<s> If this month is March, and 7 months pass, what month is it? Answer: 10 months have passed, so it is now October.</s>sports'],
['<s> If this month is August, and 9 months pass, what month is it? Answer:  If it is August and 9 months pass, then it is May.'],
['<s> If this month is October, and 3 months pass, what month is it? Answer: 3 months after October is January.</s>sports betting, sportsbook'],
['<s> If this month is April, and 1 months pass, what month is it? Answer:  If this month is April and 1 month passes, then it is May'],
['<s> If this month is November, and 12 months pass, what month is it? Answer: 12 months later, it is November again.</s>sports betting'],
['<s> If this month is September, and 9 months pass, what month is it? Answer:  If it is September and 9 months pass, then it is June.'],
['<s> If this month is July, and 9 months pass, what month is it? Answer: 9 months after July is April.\n\nIf this month is July,'],
['<s> If this month is March, and 1 months pass, what month is it? Answer:  If this month is March and 1 month passes, then it is April'],
['<s> If this month is October, and 2 months pass, what month is it? Answer:  If this month is October and 2 months pass, then it is December'],
['<s> If this month is January, and 7 months pass, what month is it? Answer: 8 months have passed, so it is now August.</s>sports bet'],
['<s> If this month is December, and 12 months pass, what month is it? Answer: 12 months have passed, so it is now December again.</s>s'],
['<s> If this month is March, and 11 months pass, what month is it? Answer: 11 months after March is February.\n\nIf this month is March'],
['<s> If this month is May, and 4 months pass, what month is it? Answer:  If this month is May and 4 months pass, then it is September'],
['<s> If this month is August, and 2 months pass, what month is it? Answer:  If it is August and 2 months pass, then it is October.'],
['<s> If this month is December, and 10 months pass, what month is it? Answer:  If this month is December and 10 months pass, then the next'],
['<s> If this month is January, and 2 months pass, what month is it? Answer: 3 months have passed, so it is now March.</s>sports bet'],
['<s> If this month is June, and 5 months pass, what month is it? Answer:  If this month is June and 5 months pass, then it is now'],
['<s> If this month is October, and 1 months pass, what month is it? Answer: 1 month has passed, so it is now November.</s>sports bet']]


# In[80]:


outputs = [out[0] for out in outputs]


# In[81]:


calculate_correct_answer_percentage(outputs, answers_of_correct_prompts)


# since output wasn't saved, it was copied into chatgpt to reformat. stupid chatgpt filled in the correct answer of these even though it wasn't told to do so

# In[102]:


indices_with_correct_answers


# In[ ]:


indices_with_correct_answers = find_indices_with_correct_answers(outputs, answers_of_correct_prompts)
correct_prompts = [correct_prompts[i] for i in indices_with_correct_answers]
answers_of_correct_prompts = [answers_of_correct_prompts[i] for i in indices_with_correct_answers]


# In[87]:


len(answers_of_correct_prompts)


# In[90]:


len(correct_prompts)


# In[91]:


# big 3 heads
head_to_remove = [(20,17), (16,0), (5,25)]
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "uno uno uno" # dos tres cinco seis

big3_outputs = []
for clean_text in correct_prompts:
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)
    big3_outputs.append(prompt_out)
    print(prompt_out)


# In[94]:


big3_outputs = [out[0] for out in big3_outputs]


# In[97]:


calculate_correct_answer_percentage(big3_outputs, answers_of_correct_prompts)


# In[98]:


# random, len 3 (not from saved head combo presets) ; ssave all results

all_outputs_all_runs = []
heads_of_circ = intersect_all
num_heads_rand = 3
num_not_overlap = len(intersect_all)
all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_of_circ] # Filter out heads_of_circ from all_possible_pairs
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "0 0 0"
for i in range(10):
    # Randomly choose pairs ensuring no overlaps with heads_of_circ
    head_to_remove = choose_heads_to_remove(filtered_pairs, heads_of_circ, num_heads_rand, num_not_overlap)
    heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]
    output_for_run = []
    for clean_text in correct_prompts:
        out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 15)
        # print(out[0])
        output_for_run.append(out[0])
    print(out)
    all_outputs_all_runs.append(output_for_run)


# In[100]:


all_scores = []
for output_for_run in all_outputs_all_runs:
    score = calculate_correct_answer_percentage(output_for_run, answers_of_correct_prompts)
    all_scores.append(score)
print(all_scores)
print(all_scores.mean())


# In[101]:


len(output_for_run)


# # If today is the Xth of month M, what date will it be in Y days?”

# In[ ]:


from datetime import datetime, timedelta
import random

def generate_prompts_and_correct_dates(N):
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    prompts = []
    correct_dates = []

    for _ in range(N):
        month_index = random.randint(0, 11)
        day = random.randint(1, 28)  # to avoid issues with different month lengths
        days_to_add = random.randint(1, 28)
        current_date = datetime(2024, month_index + 1, day)
        future_date = current_date + timedelta(days=days_to_add)
        future_month = months[future_date.month - 1]
        prompt = f"If today is {months[month_index]} {day}th, then in {days_to_add} days it will be "
        correct_date = f"{future_month} {future_date.day}th"

        prompts.append(prompt)
        correct_dates.append(correct_date)

    return prompts, correct_dates

N = 20
prompts, correct_dates = generate_prompts_and_correct_dates(N)

# Printing the results
# print("Prompts:")
# for prompt in prompts:
#     print(prompt)
# print("\nCorrect Answers:")
# for date in correct_dates:
#     print(date)


# In[ ]:


file_path = '/content/template_1_unablated_correct.txt'
correct_prompts = []
with open(file_path, 'r') as file:
    for line in file:
        correct_prompts.append([line.strip()])
print(correct_prompts)


# In[ ]:


# unablated

outputs = []
instruction = "Be concise. "
# for clean_text in correct_prompts:
for clean_text in prompts:
    # clean_text = instruction + clean_text
    # clean_text = clean_text[0]
    corr_text = "uno uno uno" # dos tres cinco seis
    heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
    mlps_not_ablate = [layer for layer in range(32)]
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
    outputs.append(prompt_out)
    print(prompt_out)


# In[ ]:


from google.colab import files
with open('template_1_unablated_wAns.txt', 'w') as f:
    for line in outputs:
        f.write(f"{line[0]}\n")
files.download('template_1_unablated_wAns.txt')


# In[ ]:


file_path = '/content/template_1_unablated_wAns.txt'
outputs = []
with open(file_path, 'r') as file:
    for line in file:
        if line != '\n':
            outputs.append(line.replace('\n', ''))
print(outputs)


# In[ ]:


outputs[-6]


# In[ ]:


correct_dates[-6]


# In[ ]:


out_ans = outputs[0].split(' ')[-2] + ' ' + outputs[0].split(' ')[-1]
out_ans = out_ans.replace('.','')
out_ans


# In[ ]:


correct_dates[0]


# In[ ]:


outputs[0].split(' ')[-2]


# In[ ]:


len(outputs)


# In[ ]:


def validate_prompt(output, correct_date):
    # try:
    #     date_str = prompt.split(" is ")[1].split(", then in ")[0]
    #     days_to_add = int(prompt.split(" in ")[1].split(" days")[0])
    #     start_date = datetime.strptime(date_str, "%B %dth")
    #     end_date = start_date + timedelta(days=days_to_add)
    #     expected_date = datetime.strptime(correct_date, "%B %dth")
    #     return end_date == expected_date
    # except ValueError:
    #     return False
    out_ans = output.split(' ')[-2] + ' ' + output.split(' ')[-1]
    out_ans = out_ans.replace('.','')
    if out_ans == correct_date:
        return True
    else:
        return False

def get_correct_prompts(outputs, correct_dates):
    corr_prompts = []
    correct_dates_of_correct_prompts = []
    for output, correct_date in zip(outputs, correct_dates):
        out_ans = output.split(' ')[-2] + ' ' + output.split(' ')[-1]
        out_ans = out_ans.replace('.','')
        if out_ans == correct_date:
            corr_prompts.append(output)
            correct_dates_of_correct_prompts.append(correct_date)
    return corr_prompts, correct_dates_of_correct_prompts

# Validate all prompts
# results = [validate_prompt(prompt, correct_date) for prompt, correct_date in zip(outputs, correct_dates)]
correctPrompts, correct_dates_of_correct_prompts = get_correct_prompts(outputs, correct_dates)

# Print the results
# for prompt, is_correct in zip(outputs, results):
#     print(f"Prompt: {prompt} - {'Correct' if is_correct else 'Incorrect'}")

# Calculate the percentage of correct prompts
# percentage_correct = sum(results) / len(results) * 100
percentage_correct = len(correctPrompts) / len(outputs) * 100
print(f"Percentage of correct prompts: {percentage_correct}%")


# In[ ]:


correctPrompts = [out.replace('<s> ', '') for out in correctPrompts]


# In[ ]:


correctPrompts = [' '.join(out.split(' ')[:-2])[:-1] for out in correctPrompts]


# In[ ]:


with open('template_1_unablated_correct.txt', 'w') as f:
    for line in correctPrompts:
        f.write(f"{line}\n")
files.download('template_1_unablated_correct.txt')


# In[ ]:


file_path = '/content/template_1_unablated_correct.txt'
correct_prompts = []
with open(file_path, 'r') as file:
    for line in file:
        correct_prompts.append(line.replace('\n', ''))
print(correct_prompts)


# In[ ]:


# intersect_all
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
head_to_remove = intersect_all
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "uno uno uno" # dos tres cinco seis

outputs = []
instruction = "Be concise. "
for clean_text in correct_prompts:
    clean_text = instruction + clean_text
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
    outputs.append(prompt_out)
    print(prompt_out)

outputs = [' '.join(out[0].replace('<s> ', '').split(' ')[:-2])[:-1] for out in outputs]
with open('template_1_intersectAll.txt', 'w') as f:
    for line in outputs:
        f.write(f"{line}\n")
files.download('template_1_intersectAll.txt')


# In[ ]:


# big 3 heads
head_to_remove = [(20,17), (16,0), (5,25)]
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "uno uno uno" # dos tres cinco seis

outputs = []
instruction = "Be concise. "
for clean_text in correct_prompts:
    clean_text = instruction + clean_text
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
    outputs.append(prompt_out)
    print(prompt_out)


# In[ ]:


outputs = [out[0] for out in outputs]
with open('template_1_big3.txt', 'w') as f:
    for line in outputs:
        f.write(f"{line}\n")
files.download('template_1_big3.txt')


# In[ ]:


file_path = '/content/template_1_big3.txt'
big_3_outputs = []
with open(file_path, 'r') as file:
    for line in file:
        if line != '\n':
            big_3_outputs.append(line.replace('\n', ''))
print(big_3_outputs)


# In[ ]:


len(big_3_outputs)


# In[ ]:


len(correct_dates_of_correct_prompts)


# In[ ]:


def get_correct_prompts_only(outputs, correct_dates):
    corr_prompts = []
    # correct_dates_of_correct_prompts = []
    for output, correct_date in zip(outputs, correct_dates):
        out_ans = output.split(' ')[-2] + ' ' + output.split(' ')[-1]
        out_ans = out_ans.replace('.','')
        if out_ans == correct_date:
            corr_prompts.append(output)
            # correct_dates_of_correct_prompts.append(correct_date)
    return corr_prompts


# In[ ]:


correctPrompts = get_correct_prompts_only(big_3_outputs, correct_dates_of_correct_prompts)
percentage_correct = len(correctPrompts) / len(big_3_outputs) * 100
print(f"Percentage of correct prompts: {percentage_correct}%")


# In[ ]:


# big 3 heads
head_to_remove = [(20,17), (16,0), (5,25)]
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "uno uno uno" # dos tres cinco seis

outputs = []
instruction = "Be concise. "
for clean_text in correct_prompts:
    clean_text = instruction + clean_text
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
    outputs.append(prompt_out)
    print(prompt_out)


# In[ ]:


# random, len 4 (not from saved head combo presets) ; ssave all results

all_prompt_outputs = []
heads_of_circ = intersect_all
num_heads_rand = 4
num_not_overlap = len(intersect_all)
all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_of_circ] # Filter out heads_of_circ from all_possible_pairs
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "0 0 0"
for clean_text in correct_prompts:
    output_for_a_prompt = []
    for i in range(10):
        # Randomly choose pairs ensuring no overlaps with heads_of_circ
        head_to_remove = choose_heads_to_remove(filtered_pairs, heads_of_circ, num_heads_rand, num_not_overlap)
        heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]
        out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
        # print(out[0])
        output_for_a_prompt.append(out[0])
    print(out)
    all_prompt_outputs.append(output_for_a_prompt)


# In[ ]:


import pdb
def num_correct_prompts_rand(list_of_list_outputs, correct_dates):
    all_scores_for_prompts = []
    for i, prompt_outputs in enumerate(list_of_list_outputs):
        num_corr_for_prompt = 0
        correct_date = correct_dates[i]
        for run_output in prompt_outputs:
            # pdb.set_trace()
            run_output = run_output[0]
            out_ans = run_output.split(' ')[-2] + ' ' + run_output.split(' ')[-1]
            # print(out_ans)
            out_ans = out_ans.replace('.','')
            if out_ans == correct_date:
                num_corr_for_prompt += 1
        perc_corr_for_prompt = num_corr_for_prompt / len(prompt_outputs)
        print(run_output, ' : ', perc_corr_for_prompt)
        all_scores_for_prompts.append(perc_corr_for_prompt)
    return sum(all_scores_for_prompts) / len(all_scores_for_prompts)  * 100


# In[ ]:


percentage_correct = num_correct_prompts_rand(all_prompt_outputs, correct_dates_of_correct_prompts) # randAbl_prompt_outputs
print(f"Percentage of correct prompts: {percentage_correct}%")


# In[ ]:


with open('template_1_rand.txt', 'w') as f:
    for line in all_prompt_outputs:
        f.write(f"{line}\n")
files.download('template_1_rand.txt')


# # (more data, runs) If today is the Xth of month M, what date will it be in Y days?”

# In[ ]:


from datetime import datetime, timedelta
import random

def generate_prompts_and_correct_dates(N):
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    prompts = []
    correct_dates = []

    for _ in range(N):
        month_index = random.randint(0, 11)
        day = random.randint(1, 28)  # to avoid issues with different month lengths
        days_to_add = random.randint(1, 28)
        current_date = datetime(2024, month_index + 1, day)
        future_date = current_date + timedelta(days=days_to_add)
        future_month = months[future_date.month - 1]
        prompt = f"If today is {months[month_index]} {day}th, then in {days_to_add} days it will be "
        correct_date = f"{future_month} {future_date.day}th"

        prompts.append(prompt)
        correct_dates.append(correct_date)

    return prompts, correct_dates

N = 100
prompts, correct_dates = generate_prompts_and_correct_dates(N)

# Printing the results
# print("Prompts:")
# for prompt in prompts:
#     print(prompt)
# print("\nCorrect Answers:")
# for date in correct_dates:
#     print(date)


# In[ ]:


# unablated

outputs = []
instruction = "Be concise. "
# for clean_text in correct_prompts:
for clean_text in prompts:
    # clean_text = instruction + clean_text
    # clean_text = clean_text[0]
    corr_text = "uno uno uno" # dos tres cinco seis
    heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
    mlps_not_ablate = [layer for layer in range(32)]
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
    outputs.append(prompt_out[0])
    print(prompt_out)


# In[ ]:


from google.colab import files


# In[ ]:


# with open('template_1_unablated_wAns.txt', 'w') as f:
#     for line in outputs:
#         f.write(f"{line[0]}\n")
# files.download('template_1_unablated_wAns.txt')

# with open('template_1_unablated_wAns.pkl', 'wb') as file:
#     pickle.dump(outputs, file)
#     files.download('template_1_unablated_wAns.pkl')


# In[ ]:


file_path = '/content/template_1_unablated_wAns.pkl'
with open(file_path, 'rb') as file:
    outputs = pickle.load(file)


# In[ ]:


outputs = [out[0] for out in outputs]


# In[ ]:


len(outputs)


# In[ ]:


out_ans = outputs[0].split(' ')[-2] + ' ' + outputs[0].split(' ')[-1]
out_ans = out_ans.replace('.','')
out_ans


# In[ ]:


correct_dates[0]


# In[ ]:


outputs[0].split(' ')[-2]


# In[ ]:


def get_correct_prompts(outputs, correct_dates):
    corr_prompts = []
    correct_dates_of_correct_prompts = []
    for output, correct_date in zip(outputs, correct_dates):
        out_ans = output.split(' ')[-2] + ' ' + output.split(' ')[-1]
        out_ans = out_ans.replace('.','')
        if out_ans == correct_date:
            corr_prompts.append(output)
            correct_dates_of_correct_prompts.append(correct_date)
    return corr_prompts, correct_dates_of_correct_prompts


# In[ ]:


correctPrompts, correct_dates_of_correct_prompts = get_correct_prompts(outputs, correct_dates)
percentage_correct = len(correctPrompts) / len(outputs) * 100
print(f"Percentage of correct prompts: {percentage_correct}%")


# In[ ]:


correct_prompts = [out.replace('<s> ', '') for out in correctPrompts]
# correct_prompts = [' '.join(out.split(' ')[:-2])[:-1] for out in correctPrompts] # keep space at end
correct_prompts = [' '.join(out.split(' ')[:-2]) for out in correct_prompts]


# In[ ]:


with open('template_1_unablated_correct.txt', 'w') as f:
    for line in correct_prompts:
        f.write(f"{line}\n")
files.download('template_1_unablated_correct.txt')

with open('template_1_unablated_correct.pkl', 'wb') as file:
    pickle.dump(correct_prompts, file)
    files.download('template_1_unablated_correct.pkl')


# In[ ]:


with open('correct_dates_of_correct_prompts.pkl', 'wb') as file:
    pickle.dump(correct_dates_of_correct_prompts, file)
    files.download('correct_dates_of_correct_prompts.pkl')


# Double check

# In[ ]:


# unablated

outputs = []
# instruction = "Be concise. "
for clean_text in correct_prompts:
    # clean_text = instruction + clean_text
    # clean_text = clean_text[0]
    corr_text = "uno uno uno" # dos tres cinco seis
    heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
    mlps_not_ablate = [layer for layer in range(32)]
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
    outputs.append(prompt_out[0])
    print(prompt_out)


# In[ ]:


percentage_correct = get_correct_prompts_only(outputs, correct_dates_of_correct_prompts)


# In[ ]:


# big 3 heads
head_to_remove = [(20,17), (16,0), (5,25)]
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "uno uno uno" # dos tres cinco seis

big_3_outputs = []
# instruction = "Be concise. "
for clean_text in correct_prompts:
    # clean_text = instruction + clean_text
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
    big_3_outputs.append(prompt_out[0])
    print(prompt_out)


# In[ ]:


def get_correct_prompts_only(outputs, correct_dates):
    num_corr_prompts = 0
    for output, correct_date in zip(outputs, correct_dates):
        out_ans = output.split(' ')[-2] + ' ' + output.split(' ')[-1]
        out_ans = out_ans.replace('.','')
        if out_ans == correct_date:
            num_corr_prompts += 1
    percentage_correct = num_corr_prompts / len(outputs) * 100
    print(f"Percentage of correct prompts: {percentage_correct}%")
    return percentage_correct


# In[ ]:


percentage_correct = get_correct_prompts_only(big_3_outputs, correct_dates_of_correct_prompts)


# In[ ]:


with open('template_1_big3.txt', 'w') as f:
    for line in big_3_outputs:
        f.write(f"{line}\n")
files.download('template_1_big3.txt')

with open('template_1_big3.pkl', 'wb') as file:
    pickle.dump(big_3_outputs, file)
    files.download('template_1_big3.pkl')


# In[ ]:


# big 5 heads
head_to_remove = [(20,17), (16,0), (5,25), (6,11), (11,18)]
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
heads_not_ablate = [x for x in heads_not_ablate if (x not in head_to_remove)]
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "uno uno uno" # dos tres cinco seis

big5_outputs = []
# instruction = "Be concise. "
for i, clean_text in enumerate(correct_prompts):
    # clean_text = instruction + clean_text
    prompt_out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
    big5_outputs.append(prompt_out[0])
    print(i, prompt_out)


# In[ ]:


percentage_correct = get_correct_prompts_only(big5_outputs, correct_dates_of_correct_prompts)


# In[ ]:


with open('template_1_big5.txt', 'w') as f:
    for line in big5_outputs:
        f.write(f"{line}\n")
files.download('template_1_big5.txt')

with open('template_1_big5.pkl', 'wb') as file:
    pickle.dump(big5_outputs, file)
    files.download('template_1_big5.pkl')


# In[ ]:


# random, len 3 (not from saved head combo presets) ; ssave all results

all_prompt_outputs = []
heads_of_circ = intersect_all
num_heads_rand = 3
num_not_overlap = len(intersect_all)
all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_of_circ] # Filter out heads_of_circ from all_possible_pairs
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "0 0 0"
for p, clean_text in enumerate(correct_prompts):
    output_for_a_prompt = []
    for i in range(10):
        # Randomly choose pairs ensuring no overlaps with heads_of_circ
        head_to_remove = choose_heads_to_remove(filtered_pairs, heads_of_circ, num_heads_rand, num_not_overlap)
        heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]
        out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
        # print(out[0])
        output_for_a_prompt.append(out[0])
    print(p, out)
    all_prompt_outputs.append(output_for_a_prompt)


# In[ ]:


import pdb
def num_correct_prompts_rand(list_of_list_outputs, correct_dates):
    all_scores_for_prompts = []
    for i, prompt_outputs in enumerate(list_of_list_outputs):
        num_corr_for_prompt = 0
        correct_date = correct_dates[i]
        for run_output in prompt_outputs:
            # pdb.set_trace()
            # run_output = run_output[0]
            out_ans = run_output.split(' ')[-2] + ' ' + run_output.split(' ')[-1]
            # print(out_ans)
            out_ans = out_ans.replace('.','')
            if out_ans == correct_date:
                num_corr_for_prompt += 1
        perc_corr_for_prompt = num_corr_for_prompt / len(prompt_outputs)
        print(run_output, ' : ', perc_corr_for_prompt)
        all_scores_for_prompts.append(perc_corr_for_prompt)
    return sum(all_scores_for_prompts) / len(all_scores_for_prompts)  * 100


# In[ ]:


percentage_correct = num_correct_prompts_rand(all_prompt_outputs, correct_dates_of_correct_prompts) # randAbl_prompt_outputs
print(f"Percentage of correct prompts: {percentage_correct}%")


# In[ ]:


with open('template_1_rand.txt', 'w') as f:
    for line in all_prompt_outputs:
        f.write(f"{line}\n")
files.download('template_1_rand.txt')

with open('template_1_rand.pkl', 'wb') as file:
    pickle.dump(all_prompt_outputs, file)
    files.download('template_1_rand.pkl')


# In[ ]:


# random, len 3 (not from saved head combo presets) ; ssave all results

all_prompt_outputs = []
heads_of_circ = intersect_all
num_heads_rand = 3
num_not_overlap = len(intersect_all)
all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_of_circ] # Filter out heads_of_circ from all_possible_pairs
mlps_not_ablate = [layer for layer in range(32)]
corr_text = "0 0 0"
for p, clean_text in enumerate(correct_prompts):
    output_for_a_prompt = []
    for i in range(50):
        # Randomly choose pairs ensuring no overlaps with heads_of_circ
        head_to_remove = choose_heads_to_remove(filtered_pairs, heads_of_circ, num_heads_rand, num_not_overlap)
        heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]
        out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)
        # print(out[0])
        output_for_a_prompt.append(out[0])
    print(p, out)
    all_prompt_outputs.append(output_for_a_prompt)


# In[ ]:


percentage_correct = num_correct_prompts_rand(all_prompt_outputs[:-1], correct_dates_of_correct_prompts) # randAbl_prompt_outputs
print(f"Percentage of correct prompts: {percentage_correct}%")


# # What are the months in a year?

# In[ ]:


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
        # if i == corr_ans_tokLen - 1:
        #     print(model.to_string(tokens))
            # print(f"Sequence so far: {clean_text}")
            # print(f"{tokens.shape[-1]+1}th char = {next_char!r}")

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
    return model.to_string(tokens)


# In[ ]:


clean_text = "What are the months in a year? Give all of them as a list. Be concise."
corr_text = "5 3 9"
num_toks_gen = 50


# In[ ]:


output = []
for i in range(50):
    heads_of_circ = nums_1to9
    num_heads_rand = 86
    num_not_overlap = len(nums_1to9)

    all_possible_pairs =  [(layer, head) for layer in range(32) for head in range(32)]
    # Filter out heads_of_circ from all_possible_pairs
    filtered_pairs = [pair for pair in all_possible_pairs if pair not in heads_of_circ]

    # Randomly choose 100 pairs ensuring less than 50 overlaps with heads_of_circ
    head_to_remove = choose_heads_to_remove(filtered_pairs, heads_of_circ, 86, num_not_overlap)

    heads_not_ablate = [x for x in all_possible_pairs if x not in head_to_remove]

    mlps_not_ablate = [layer for layer in range(32)]

    out = ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, num_toks_gen)
    print(out)
    output.append(out)


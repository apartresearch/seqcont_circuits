#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


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


# # load generate fn kwargs

# In[ ]:


# Get list of arguments to pass to `generate` (specifically these are the ones relating to sampling)
generate_kwargs = dict(
    do_sample = False, # deterministic output so we can compare it to the HF model
    top_p = 1.0, # suppresses annoying output errors
    temperature = 1.0, # suppresses annoying output errors
)


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
        if i == corr_ans_tokLen - 1:
            print(model.to_string(tokens))
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


# # succeeds on

# ## word problems

# In[ ]:


instruction = "Be concise. "
clean_text =  "If today is the 11th of a month, what date will it be in 6 days?"
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 10)


# In[ ]:


instruction = "Be concise. "
clean_text =  "What number comes after 3?"
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 10)


# In[ ]:


instruction = "Be concise. "
clean_text =  "What number comes after 3002? Answer: "
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# In[ ]:


instruction = "Be concise. "
clean_text =  "If this month is September, and 3 months pass, what is month name is it? Answer: December. If this month is March, and 3 months pass, what month name is it? Answer: "
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# In[ ]:


instruction = "Be concise. "
clean_text =  "What are all the months in Fall? List them in order."
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 30)


# In[ ]:


instruction = "Answer yes or no. "
clean_text =  "Is 16 greater than 11? Answer: "
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[ ]:


instruction = "Be concise. "
clean_text =  "What number is greater than 11? Answer: "
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# ## arithmetic

# In[ ]:


# instruction = "Be concise. "
clean_text =  "5 + 16 = "
# clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[ ]:


instruction = "Be concise. "
clean_text =  "100 + 58 = "
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# In[ ]:


instruction = "Be concise. "
clean_text =  "5 x 6 = "
# clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# In[ ]:


instruction = "Be concise. "
clean_text =  "2 x 2 = "
# clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 3)


# # fails on

# In[ ]:


clean_text = "Given the sequence 2, 4, 6, 8, ..., identify the 10th term. Be concise."
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 10)


# In[ ]:


instruction = "Be concise. "
clean_text =  "If this month is March, and 3 months pass, what month name is it? Answer: "
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 5)


# In[ ]:


clean_text = "Two days after Monday is"
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 20)


# In[ ]:


clean_text = "Be concise. In the arithmetic sequence starting at 5 and increasing by 3 each time, what is the 7th term?"
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 20)


# In[ ]:


instruction = "Be concise. "
clean_text =  "Starting on the 1st of January, what date falls 50 days later?"
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 20)


# In[ ]:


# Be concise. In a cyclic pattern of colors: Red, Blue, Green, Yellow, what color comes after the 3rd color?


# # succeeds on- spanish

# In[ ]:


instruction = "Be concise. "
clean_text =  "Qué viene después de uno?"
# clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 30)


# In[ ]:


prompt = "Qué viene después de uno?"
output = model.generate(prompt, max_new_tokens=20, **generate_kwargs)
print(output)


# In[ ]:


instruction = "Sé conciso. "
clean_text =  "Qué viene después de uno?"
clean_text = instruction + clean_text
corr_text = "uno uno uno" # dos tres cinco seis
heads_not_ablate = [(layer, head) for layer in range(32) for head in range(32)]  # unablated
mlps_not_ablate = [layer for layer in range(32)]
ablate_then_gen(model, clean_text, corr_text, heads_not_ablate, mlps_not_ablate, 20)


# In[ ]:


prompt = "Sé conciso. Qué viene después de uno?"
output = model.generate(prompt, max_new_tokens=30, **generate_kwargs)
print(output)


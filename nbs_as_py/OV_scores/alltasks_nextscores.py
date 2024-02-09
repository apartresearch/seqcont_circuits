# -*- coding: utf-8 -*-
"""allTasks_nextScores.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CSoW6kEjS8ByTtxyUCSwQtaShd3uhQtn

<a href="https://colab.research.google.com/github/wlg100/numseqcont_circuit_expms/blob/main/notebook_templates/headFNs_expms_template.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" align="left"/></a>&nbsp;or in a local notebook.

# Setup
"""

# %pip install git+https://github.com/neelnanda-io/TransformerLens.git

# import torch
# import transformer_lens
# import transformer_lens.utils as utils
# from transformer_lens.hook_points import (
#     HookedRootModule,
#     HookPoint,
# )  # Hooking utilities
# from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
# torch.set_grad_enabled(False)

# model_name = "gpt2-small"
# model = HookedTransformer.from_pretrained(
#     model_name,
#     center_unembed=True,
#     center_writing_weights=True,
#     fold_ln=True,
#     refactor_factored_attn_matrices=True,
# )

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# %pip install git+https://github.com/redwoodresearch/Easy-Transformer.git
# %pip install einops datasets transformers fancy_einsum

# assert torch.cuda.device_count() == 1
# from tqdm import tqdm
# import pandas as pd
import torch
import torch as t
from easy_transformer.EasyTransformer import (
    EasyTransformer,
)
# from time import ctime
# from functools import partial

import numpy as np
import einops
from copy import deepcopy

model = EasyTransformer.from_pretrained("gpt2").cuda()
# model = EasyTransformer.from_pretrained("gpt2")
model.set_use_attn_result(True)

"""# Dataset and Score Functions"""

class Dataset:
    def __init__(self, prompts, tokenizer, S1_is_first=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.N = len(prompts)

        texts = [ prompt["text"] for prompt in self.prompts ]
        self.toks = torch.Tensor(self.tokenizer(texts, padding=True).input_ids).type(
            torch.int
        )

        # word_idx: for every prompt, find the token index of each target token and "end"
        # word_idx is a tensor with an element for each prompt. The element is the targ token's ind at that prompt
        self.word_idx = {}
        for targ in [key for key in self.prompts[0].keys() if key != 'text']:
            targ_lst = []
            for prompt in self.prompts:
                input_text = prompt["text"]
                tokens = model.tokenizer.tokenize(input_text)
                if S1_is_first and targ == "S1":  # only use this if first token doesn't have space Ġ in front
                    target_token = prompt[targ]
                else:
                    target_token = "Ġ" + prompt[targ]
                target_index = tokens.index(target_token)
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

def get_next_scores(model, layer, head, dataset, task="numerals", neg=False, print_all_results=True):
    cache = {}
    model.cache_some(cache, lambda x: x == "blocks.0.hook_resid_post")
    model(dataset.toks.long())
    if neg:
        sign = -1
    else:
        sign = 1
    z_0 = model.blocks[1].attn.ln1(cache["blocks.0.hook_resid_post"])
    # z_0 = model.blocks[1].attn.hook_z(cache["blocks.0.hook_resid_post"])

    v = torch.einsum("eab,bc->eac", z_0, model.blocks[layer].attn.W_V[head])
    v += model.blocks[layer].attn.b_V[head].unsqueeze(0).unsqueeze(0)

    o = sign * torch.einsum("sph,hd->spd", v, model.blocks[layer].attn.W_O[head])
    logits = model.unembed(model.ln_final(o))

    k = 5
    n_right = 0

    pred_tokens_dict = {}
    words_moved = []
    # get the keys from the first prompt in the dataset
    words = [key for key in dataset.prompts[0].keys() if key != 'text']

    numwords = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    ranks = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth']

    for seq_idx, prompt in enumerate(dataset.prompts):
        for word in words:
            pred_tokens = [
                model.tokenizer.decode(token)
                for token in torch.topk(
                    logits[seq_idx, dataset.word_idx[word][seq_idx]], k
                ).indices
            ]

            # get next member after seq member prompt[word]
            if task == "numerals":
                next_word = str(int(prompt[word]) + 1)
            elif task == "numwords":
                next_word = str(numwords[numwords.index(prompt[word]) + 1])
            elif task == "months":
                next_word = str(ranks[months.index(prompt[word]) + 1])

            nextToken_in_topK = 'no'
            if " " + next_word in pred_tokens or next_word in pred_tokens:
                n_right += 1
                words_moved.append(prompt[word])
                nextToken_in_topK = 'yes'
            pred_tokens_dict[prompt[word]] = (pred_tokens, next_word, nextToken_in_topK)

    percent_right = (n_right / (dataset.N * len(words))) * 100
    if percent_right > 0:
        print(f"Next circuit for head {layer}.{head}: Top {k} accuracy: {percent_right}%")

    if print_all_results == True:
        print(pred_tokens_dict)

    return percent_right

def get_copy_scores(model, layer, head, dataset, neg=False, print_all_results=True):
    cache = {}
    model.cache_some(cache, lambda x: x == "blocks.0.hook_resid_post")
    model(dataset.toks.long())
    if neg:
        sign = -1
    else:
        sign = 1
    z_0 = model.blocks[1].attn.ln1(cache["blocks.0.hook_resid_post"])
    # z_0 = model.blocks[1].attn.hook_z(cache["blocks.0.hook_resid_post"])

    v = torch.einsum("eab,bc->eac", z_0, model.blocks[layer].attn.W_V[head])
    v += model.blocks[layer].attn.b_V[head].unsqueeze(0).unsqueeze(0)

    o = sign * torch.einsum("sph,hd->spd", v, model.blocks[layer].attn.W_O[head])
    logits = model.unembed(model.ln_final(o))

    k = 5
    n_right = 0

    pred_tokens_dict = {}
    words_moved = []
    # get the keys from the first prompt in the dataset
    words = [key for key in dataset.prompts[0].keys() if key != 'text']

    for seq_idx, prompt in enumerate(dataset.prompts):
        for word in words:
            pred_tokens = [
                model.tokenizer.decode(token)
                for token in torch.topk(
                    logits[seq_idx, dataset.word_idx[word][seq_idx]], k
                ).indices
            ]

            token_in_topK = 'no'
            if " " + prompt[word] in pred_tokens or prompt[word] in pred_tokens:
                n_right += 1
                words_moved.append(prompt[word])
                token_in_topK = 'yes'
            pred_tokens_dict[prompt[word]] = (pred_tokens, token_in_topK)

    percent_right = (n_right / (dataset.N * len(words))) * 100
    if percent_right > 0:
        print(f"Copy circuit for head {layer}.{head}: Top {k} accuracy: {percent_right}%")

    if print_all_results == True:
        print(pred_tokens_dict)

    return percent_right

"""# Numerals"""

def generate_prompts_list():
    prompts_list = []
    for i in range(1, 98):
        prompt_dict = {
            'S1': str(i),
            'S2': str(i+1),
            'S3': str(i+2),
            'S4': str(i+3),
            'text': f"{i} {i+1} {i+2} {i+3}"
        }
        prompts_list.append(prompt_dict)
    return prompts_list

prompts_list = generate_prompts_list()

dataset = Dataset(prompts_list, model.tokenizer, S1_is_first=True)

get_next_scores(model, 9, 1, dataset, task="numerals")

get_copy_scores(model, 9, 1, dataset)

# %%capture
all_copy_scores = []
all_heads = [(layer, head) for layer in range(12) for head in range(12)]
for index, (layer, head) in enumerate(all_heads):
    all_copy_scores.append(get_copy_scores(model, layer, head, dataset, print_all_results=False))
    # all_copy_scores.append( (layer, head, get_copy_scores(model, layer, head, dataset, print_all_results=False)) )

sum(all_copy_scores)/len(all_copy_scores)

# %%capture
all_next_scores = []
all_heads = [(layer, head) for layer in range(12) for head in range(12)]
for index, (layer, head) in enumerate(all_heads):
    all_next_scores.append(get_next_scores(model, layer, head, dataset, task="numerals", print_all_results=False))
    # all_next_scores.append( (layer, head, get_next_scores(model, layer, head, dataset, task="numerals", print_tokens=False)) )

sum(all_next_scores)/len(all_next_scores)

"""# Numwords"""

def generate_prompts_list(x ,y):
    words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
    prompts_list = []
    for i in range(x, y):
        prompt_dict = {
            'S1': words[i],
            'S2': words[i+1],
            'S3': words[i+2],
            'S4': words[i+3],
            'text': f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
        }
        prompts_list.append(prompt_dict)
    return prompts_list

prompts_list = generate_prompts_list(0, 8)
dataset = Dataset(prompts_list, model.tokenizer, S1_is_first=True)

all_next_scores = []
all_heads = [(layer, head) for layer in range(12) for head in range(12)]
for index, (layer, head) in enumerate(all_heads):
    all_next_scores.append(get_next_scores(model, layer, head, dataset, task="numwords", print_all_results=False))

sum(all_next_scores)/len(all_next_scores)

"""# Months"""

def generate_prompts_list():
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    prompts_list = []
    for i in range(0, 8):
        prompt_dict = {
            'S1': months[i],
            'S2': months[i+1],
            'S3': months[i+2],
            'S4': months[i+3],
            'text': f"{months[i]} {months[i+1]} {months[i+2]} {months[i+3]}"
        }
        prompts_list.append(prompt_dict)
    return prompts_list

prompts_list = generate_prompts_list()
dataset = Dataset(prompts_list, model.tokenizer, S1_is_first=True)

all_copy_scores = []
all_heads = [(layer, head) for layer in range(12) for head in range(12)]
for index, (layer, head) in enumerate(all_heads):
    all_copy_scores.append(get_copy_scores(model, layer, head, dataset, print_all_results=False))

sum(all_copy_scores)/len(all_copy_scores)

all_next_scores = []
all_heads = [(layer, head) for layer in range(12) for head in range(12)]
for index, (layer, head) in enumerate(all_heads):
    all_next_scores.append(get_next_scores(model, layer, head, dataset, task="months", print_all_results=False))

sum(all_next_scores)/len(all_next_scores)

get_next_scores(model, 9, 1, dataset, task="months")

import torch
import torch as t
from easy_transformer.EasyTransformer import (
    EasyTransformer,
)
import numpy as np
import einops
from copy import deepcopy

model = EasyTransformer.from_pretrained("gpt2").cuda()
# model = EasyTransformer.from_pretrained("gpt2")
model.set_use_attn_result(True)

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
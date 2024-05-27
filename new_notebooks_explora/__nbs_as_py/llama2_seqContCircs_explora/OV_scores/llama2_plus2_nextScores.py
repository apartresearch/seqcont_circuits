#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/wlg100/numseqcont_circuit_expms/blob/main/notebook_templates/headFNs_expms_template.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" align="left"/></a>&nbsp;or in a local notebook.

# # Setup

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install accelerate\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%pip install git+https://github.com/neelnanda-io/TransformerLens.git\n')


# In[3]:


get_ipython().run_line_magic('pip', 'install einops')


# In[4]:


import torch
import numpy as np
import einops
from copy import deepcopy


# # Load Model

# In[8]:


from transformers import LlamaForCausalLM, LlamaTokenizer


# In[9]:


get_ipython().system('huggingface-cli login')


# In[10]:


LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-chat-hf"

# tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH)
tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH, use_fast= False, add_prefix_space= False)
hf_model = LlamaForCausalLM.from_pretrained(LLAMA_2_7B_CHAT_PATH, low_cpu_mem_usage=True)


# In[12]:


import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer


# In[13]:


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


# # Dataset and Score Functions

# In[16]:


class Dataset:
    def __init__(self, prompts, tokenizer):  # , S1_is_first=False
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
        pos_dict = {}
        list_tokens = tokenizer.tokenize('2 4 6 ')
        for i, tok_as_str in enumerate(list_tokens):
            pos_dict['S'+str(i)] = i

        # word_idx: for every prompt, find the token index of each target token and "end"
        # word_idx is a tensor with an element for each prompt. The element is the targ token's ind at that prompt
        self.word_idx = {}
        # for targ in [key for key in self.prompts[0].keys() if (key != 'text' and key != 'corr' and key != 'incorr')]:
        for targ in [key for key in pos_dict]:
            targ_lst = []
            for prompt in self.prompts:
                input_text = prompt["text"]
                tokens = self.tokenizer.tokenize(input_text)
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


model.blocks[1].attn


# In[19]:


def get_copy_scores(model, layer, head, dataset, neg=False, print_all_results=True):
    cache = {}
    model.cache_some(cache, lambda x: x == "blocks.0.hook_resid_post")
    model(dataset.toks.long())
    if neg:
        sign = -1
    else:
        sign = 1
    # z_0 = model.blocks[1].attn.ln1(cache["blocks.0.hook_resid_post"])
    z_0 = model.blocks[1].attn.hook_z(cache["blocks.0.hook_resid_post"])

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


# In[18]:


def get_next_scores(model, layer, head, dataset, task="numerals", neg=False, print_all_results=True):
    cache = {}
    model.cache_some(cache, lambda x: x == "blocks.0.hook_resid_post")
    model(dataset.toks.long())
    if neg:
        sign = -1
    else:
        sign = 1
    # z_0 = model.blocks[1].attn.ln1(cache["blocks.0.hook_resid_post"])
    z_0 = model.blocks[1].attn.hook_z(cache["blocks.0.hook_resid_post"])

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


# In[20]:


def get_next_next_scores(model, layer, head, dataset, task="numerals", neg=False, print_all_results=True):
    cache = {}
    model.cache_some(cache, lambda x: x == "blocks.0.hook_resid_post")
    model(dataset.toks.long())
    if neg:
        sign = -1
    else:
        sign = 1
    # z_0 = model.blocks[1].attn.ln1(cache["blocks.0.hook_resid_post"])
    z_0 = model.blocks[1].attn.hook_z(cache["blocks.0.hook_resid_post"])

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

    # numwords = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
    # months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    # ranks = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth']

    for seq_idx, prompt in enumerate(dataset.prompts):
        for word in words:
            pred_tokens = [
                model.tokenizer.decode(token)
                for token in torch.topk(
                    logits[seq_idx, dataset.word_idx[word][seq_idx]], k
                ).indices
            ]

            # get next NEXT member after seq member prompt[word]
            if task == "numerals":
                next_word = str(int(prompt[word]) + 2)
            # elif task == "numwords":
            #     next_word = str(numwords[numwords.index(prompt[word]) + 1])
            # elif task == "months":
            #     next_word = str(ranks[months.index(prompt[word]) + 1])

            nextToken_in_topK = 'no'
            if " " + next_word in pred_tokens or next_word in pred_tokens:
                n_right += 1
                words_moved.append(prompt[word])
                nextToken_in_topK = 'yes'
            pred_tokens_dict[prompt[word]] = (pred_tokens, next_word, nextToken_in_topK)

    percent_right = (n_right / (dataset.N * len(words))) * 100
    if percent_right > 0:
        print(f"Next next score for head {layer}.{head}: Top {k} accuracy: {percent_right}%")

    if print_all_results == True:
        print(pred_tokens_dict)

    return percent_right


# # Numerals

# ## make dataset

# In[21]:


def generate_prompts():
    prompts_list = []
    # prompt_dict = {
    #     # 'corr': '8',
    #     # 'incorr': '6',
    #     'text': f"2 4 6 "
    # }
    # list_tokens = tokenizer.tokenize('2 4 6 ')
    # for i, tok_as_str in enumerate(list_tokens):
    #     if tok_as_str == '▁':
    #         prompt_dict['S'+str(i)] = ' '
    #     else:
    #         prompt_dict['S'+str(i)] = tok_as_str
    prompt_dict = {
        'S1': '2',
        'S2': '4',
        'S3': '6',
        'text': f"2 4 6 "
    }
    prompts_list.append(prompt_dict)
    return prompts_list

prompts_list = generate_prompts()
prompts_list


# In[22]:


tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# In[23]:


dataset = Dataset(prompts_list, tokenizer)


# ## get copy scores

# In[ ]:


get_copy_scores(model, 16, 0, dataset)


# In[ ]:


# %%capture
# all_copy_scores = []
all_copy_scores = {}
all_heads = [(layer, head) for layer in range(32) for head in range(32)]
for index, (layer, head) in enumerate(all_heads):
    # all_copy_scores.append(get_copy_scores(model, layer, head, dataset, print_all_results=False))
    all_copy_scores[(layer, head)] = get_copy_scores(model, layer, head, dataset, print_all_results=False)


# In[ ]:


k=5
for (layer, head), percent_right in all_copy_scores.items():
    if percent_right > 0:
        print(f"Copy circuit for head {layer}.{head}: Top {k} accuracy: {percent_right}%")


# In[ ]:


sum(all_copy_scores.values())/len(all_copy_scores)


# ## get next scores

# In[ ]:


# %%capture
# all_next_scores = []
all_next_scores = {}
all_heads = [(layer, head) for layer in range(32) for head in range(32)]
for index, (layer, head) in enumerate(all_heads):
    # all_next_scores.append(get_next_scores(model, layer, head, dataset, task="numerals", print_all_results=False))
    all_next_scores[(layer, head)] = get_next_scores(model, layer, head, dataset, print_all_results=False)


# In[28]:


k=5
for (layer, head), percent_right in all_next_scores.items():
    if percent_right > 0:
        print(f"Next score for head {layer}.{head}: Top {k} accuracy: {percent_right}%")


# In[27]:


sum(all_next_scores.values())/len(all_next_scores)


# In[ ]:


get_next_scores(model, 22, 31, dataset, task="numerals")


# In[30]:


get_next_scores(model, 30, 13, dataset, task="numerals")


# ## next next scores

# In[24]:


get_next_next_scores(model, 16, 0, dataset, task="numerals")


# In[ ]:


all_next_next_scores = {}
all_heads = [(layer, head) for layer in range(32) for head in range(32)]
for index, (layer, head) in enumerate(all_heads):
    percent_right = get_next_next_scores(model, layer, head, dataset, print_all_results=False)
    if percent_right > 0:
        all_next_next_scores[(layer, head)] = percent_right


# In[31]:


k=5
for (layer, head), percent_right in all_next_next_scores.items():
    if percent_right > 0:
        print(f"Next +2 score for head {layer}.{head}: Top {k} accuracy: {percent_right}%")


# In[34]:


for (layer, head), percent_right in all_next_next_scores.items():
    if percent_right > 0:
        get_next_next_scores(model, layer, head, dataset, print_all_results=True)


# In[33]:


sum(all_next_next_scores.values())/len(all_next_next_scores)


# In[32]:


get_next_next_scores(model, 31, 30, dataset, task="numerals")


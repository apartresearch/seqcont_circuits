#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install accelerate\n')


# In[2]:


# %%capture
# %pip install git+https://github.com/neelnanda-io/TransformerLens.git


# In[3]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import torch.nn.functional as F


# In[4]:


# model_name = 'gpt2'
# device = 'cuda:0' if t.cuda.is_available() else 'cpu'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=t.float16).to(device)


# ## Load Model

# In[5]:


from transformers import LlamaForCausalLM, LlamaTokenizer


# In[6]:


get_ipython().system('huggingface-cli login')


# In[7]:


LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-chat-hf"

# tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH)
tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH, use_fast= False, add_prefix_space= False)
hf_model = LlamaForCausalLM.from_pretrained(LLAMA_2_7B_CHAT_PATH, low_cpu_mem_usage=True)


# In[9]:


# import transformer_lens.utils as utils
# from transformer_lens.hook_points import HookPoint
# from transformer_lens import HookedTransformer


# In[ ]:


# model = HookedTransformer.from_pretrained(
#     LLAMA_2_7B_CHAT_PATH,
#     hf_model = hf_model,
#     tokenizer = tokenizer,
#     device = "cpu",
#     fold_ln = False,
#     center_writing_weights = False,
#     center_unembed = False,
# )

# del hf_model

# model = model.to("cuda" if t.cuda.is_available() else "cpu")


# In[10]:


device = 'cuda:0' if t.cuda.is_available() else 'cpu'
# model = model.to(device)
model = hf_model.to(device)


# # test code

# In[ ]:


class LlamaForCausalLMWithLogitLens(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask=None, labels=None, output_attentions=None, output_hidden_states=True, return_dict=True):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Extract hidden states from all layers
        hidden_states = outputs.hidden_states

        # Compute logits for each layer
        logits_per_layer = []
        for layer_hidden_state in hidden_states:
            logits = self.lm_head(layer_hidden_state)
            logits_per_layer.append(logits)

        return outputs, logits_per_layer

# Initialize the modified model
model_with_logit_lens = LlamaForCausalLMWithLogitLens.from_pretrained(LLAMA_2_7B_CHAT_PATH, low_cpu_mem_usage=True)


# In[ ]:


def get_logits_per_layer(input_text):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs['input_ids']

    # Forward pass through the model
    with t.no_grad():
        outputs, logits_per_layer = model_with_logit_lens(input_ids)

    return logits_per_layer

# Example usage
input_text = "This is an example sentence."
logits_per_layer = get_logits_per_layer(input_text)


# # Functions

# In[16]:


def actvs_to_logits(hidden_states):
    """
    outputs.hidden_states is a tuple for every layer
    each tuple member is an actvs tensor of size (batch_size, seq_len, d_model)
    loop thru tuple to get actv for each layer
    """
    layer_logits_list = []  # logits for each layer hidden state output actvs
    for i, h in enumerate(hidden_states):
        h_last_tok = h[:, -1, :]
        # if i == len(hidden_states) - 1:
        #     h_last_tok = model.transformer.ln_f(h_last_tok)  # apply layer norm as not in last
        logits = t.einsum('ab, cb -> ac', model.lm_head.weight, h_last_tok)
        layer_logits_list.append(logits)
    return layer_logits_list

def get_logits(input_text):
    token_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = model(token_ids, output_hidden_states=True)
    logits = actvs_to_logits(outputs.hidden_states)
    logits = t.stack(logits).squeeze(-1)
    return logits


# In[17]:


def get_decoded_indiv_toks(layer_logits, k=10):
    """
    i is the layer (from before to last).
    layer_logits[i] are the scores for each token in vocab dim for the ith unembedded layer
    j is the top 5
    """
    output_list = []
    for i, layer in enumerate(layer_logits):
        top_5_at_layer = []
        sorted_token_ids = F.softmax(layer_logits[i],dim=-1).argsort(descending=True)
        for j in range(5):  # loop to separate them in a list, rather than concat into one str
            top_5_at_layer.append(tokenizer.decode(sorted_token_ids[j]))
        output_list.append( top_5_at_layer )
    return output_list


# # Test one samp, diff intervals

# In[19]:


prompts = ["1 2 3 ", "2 4 6 "]
for test_text in prompts:
    layer_logits = get_logits(test_text)
    tok_logit_lens = get_decoded_indiv_toks(layer_logits)
    for i, tokouts in enumerate(tok_logit_lens):
        print(i-1, tokouts)
    print('\n')


# In[20]:


prompts = ["0 2 4 "]
for test_text in prompts:
    layer_logits = get_logits(test_text)
    tok_logit_lens = get_decoded_indiv_toks(layer_logits)
    for i, tokouts in enumerate(tok_logit_lens):
        print(i-1, tokouts)
    print('\n')


# In[21]:


prompts = ["2 4 6 8 "]
for test_text in prompts:
    layer_logits = get_logits(test_text)
    tok_logit_lens = get_decoded_indiv_toks(layer_logits)
    for i, tokouts in enumerate(tok_logit_lens):
        print(i-1, tokouts)
    print('\n')


# In[23]:


prompts = ["0 3 6 "]
for test_text in prompts:
    layer_logits = get_logits(test_text)
    tok_logit_lens = get_decoded_indiv_toks(layer_logits)
    for i, tokouts in enumerate(tok_logit_lens):
        print(i-1, tokouts)
    print('\n')


# In[22]:


prompts = ["3 6 9 "]
for test_text in prompts:
    layer_logits = get_logits(test_text)
    tok_logit_lens = get_decoded_indiv_toks(layer_logits)
    for i, tokouts in enumerate(tok_logit_lens):
        print(i-1, tokouts)
    print('\n')


# # pure seq prompts

# In[ ]:


def generate_prompts_list(x ,y):
    prompts_list = []
    for i in range(x, y):
        prompt_dict = {
            'corr': str(i+4),
            'incorr': str(i+3),
            'text': f"{i} {i+2} {i+4} {i+3}"
        }
        prompts_list.append(prompt_dict)
    return prompts_list

prompts_list = generate_prompts_list(1, 9)


# In[ ]:


# for pd in prompts_list:
#     test_text = pd['text']
#     layer_logits = get_logits(test_text)
#     tok_logit_lens = get_decoded_indiv_toks(layer_logits)
#     for i, tokouts in enumerate(tok_logit_lens):
#         print(i-1, tokouts)
#     print('\n')


# # numerals 1536

# In[ ]:


task = "digits"
prompts_list = []

temps = ['done', 'lost', 'names']

for i in temps:
    file_name = f'/content/{task}_prompts_{i}.pkl'
    with open(file_name, 'rb') as file:
        filelist = pickle.load(file)

    print(filelist[0]['text'])
    prompts_list += filelist [:512] #768 512

len(prompts_list)


# In[ ]:


num_corr = 0
anomolies = []
for pd in prompts_list:
    test_text = pd['text']
    layer_logits = get_logits(test_text)
    tok_logit_lens = get_decoded_indiv_toks(layer_logits)

    """
    Check if the 8th layer's predicted token is the sequence member just "one before"
    the correct next sequence member output found in the ninth layer

    Use `try` because when indexing, the output may not be a seq member of the right type!
    """
    try:
        a = tok_logit_lens[9][0].replace(' ', '')
        b= tok_logit_lens[10][0].replace(' ', '')
        if int(a) < int(b):
            if tok_logit_lens[10][0] == pd['corr']:
                num_corr += 1
            else:
                anomolies.append(pd)
        else:
                anomolies.append(pd)
    except:
        anomolies.append(pd)
    # for i, tokouts in enumerate(tok_logit_lens):
    #     print(i-1, tokouts)
    # print('\n')


# In[ ]:


num_corr


# Do a quick scan of what prompts are anomolies in which 8th layer output is not just "the seq member one before" the 9th layer output.

# In[ ]:


for pd in anomolies[:2]:
    test_text = pd['text']
    layer_logits = get_logits(test_text)
    tok_logit_lens = get_decoded_indiv_toks(layer_logits)
    for i, tokouts in enumerate(tok_logit_lens):
        print(i-1, tokouts)
    print('\n')


# In[ ]:


for pd in anomolies[-2:]:
    test_text = pd['text']
    layer_logits = get_logits(test_text)
    tok_logit_lens = get_decoded_indiv_toks(layer_logits)
    for i, tokouts in enumerate(tok_logit_lens):
        print(i-1, tokouts)
    print('\n')


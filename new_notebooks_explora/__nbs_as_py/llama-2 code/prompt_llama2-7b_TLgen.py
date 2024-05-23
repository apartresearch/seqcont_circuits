#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer

# # Load the model and tokenizer
# model_name = "facebook/llama-2"  # Replace with the correct model name
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
# model = LlamaForCausalLM.from_pretrained(model_name)

# # Ensure the model is on the correct device (GPU if available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Function to evaluate the model on a given prompt
# def evaluate_model(prompt, max_length=50):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs.input_ids,
#             max_length=max_length,
#             num_return_sequences=1,
#             no_repeat_ngram_size=2,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95
#         )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Example prompt
# prompt = "Once upon a time"

# # Evaluate the model
# output = evaluate_model(prompt)
# print("Generated text:", output)


# # Setup

# In[1]:


get_ipython().run_cell_magic('capture', '', '%pip install git+https://github.com/neelnanda-io/TransformerLens.git\n')


# In[2]:


import torch
import numpy as np
from pathlib import Path


# In[3]:


# import transformer_lens
import transformer_lens.utils as utils
# from transformer_lens.hook_points import (
#     HookedRootModule,
#     HookPoint,
# )  # Hooking utilities
# from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache


# # Load model

# In[4]:


from transformers import LlamaForCausalLM, LlamaTokenizer


# In[5]:


get_ipython().system('huggingface-cli login')


# In[6]:


LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH)
hf_model = LlamaForCausalLM.from_pretrained(LLAMA_2_7B_CHAT_PATH, low_cpu_mem_usage=True)


# In[7]:


import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer


# In[8]:


# Get list of arguments to pass to `generate` (specifically these are the ones relating to sampling)
generate_kwargs = dict(
    do_sample = False, # deterministic output so we can compare it to the HF model
    top_p = 1.0, # suppresses annoying output errors
    temperature = 1.0, # suppresses annoying output errors
)


# In[9]:


tl_model = HookedTransformer.from_pretrained(
    LLAMA_2_7B_CHAT_PATH,
    hf_model = hf_model,
    tokenizer = tokenizer,
    device = "cpu",
    fold_ln = False,
    center_writing_weights = False,
    center_unembed = False,
)

del hf_model

tl_model = tl_model.to("cuda" if torch.cuda.is_available() else "cpu")


# # spanish

# In[10]:


prompt = "uno"
output = tl_model.generate(prompt, max_new_tokens=1, **generate_kwargs)
print(output)


# In[12]:


prompt = "uno dos"
output = tl_model.generate(prompt, max_new_tokens=1, **generate_kwargs)
print(output)


# In[11]:


prompt = "uno dos tres"
output = tl_model.generate(prompt, max_new_tokens=1, **generate_kwargs)
print(output)


# In[13]:


prompt = "uno dos tres cuatro cinco seis"
output = tl_model.generate(prompt, max_new_tokens=1, **generate_kwargs)
print(output)


# In[16]:


prompt = "uno mas tres"
output = tl_model.generate(prompt, max_new_tokens=2, **generate_kwargs)
print(output)


# In[17]:


prompt = "uno tres cinco"
output = tl_model.generate(prompt, max_new_tokens=2, **generate_kwargs)
print(output)


# # German

# In[20]:


prompt = "eins zwei drei"
output = tl_model.generate(prompt, max_new_tokens=5, **generate_kwargs)
print(output)


# In[21]:


prompt = "eins"
output = tl_model.generate(prompt, max_new_tokens=5, **generate_kwargs)
print(output)


# # French

# In[22]:


prompt = "un, deux, trois"
output = tl_model.generate(prompt, max_new_tokens=5, **generate_kwargs)
print(output)


# # math reasoning

# In[25]:


prompt = "Bob is three years older than Steve. If Bob is 6, how old is Steve?"
output = tl_model.generate(prompt, max_new_tokens=50, **generate_kwargs)
print(output)


# In[27]:


prompt = "Bob is three years older than Steve. If Bob is 6 then Steve is"
output = tl_model.generate(prompt, max_new_tokens=50, **generate_kwargs)
print(output)


# # Days of week

# In[28]:


prompt = "Today is Monday. What day comes next?"
output = tl_model.generate(prompt, max_new_tokens=50, **generate_kwargs)
print(output)


# In[43]:


prompt = "Today is Monday. In two days, it will be"
output = tl_model.generate(prompt, max_new_tokens=50, **generate_kwargs)
print(output)


# # Distinguish Additive vs Multiplicative

# In[34]:


prompt =  "2 4 6 8 10"
output = tl_model.generate(prompt, max_new_tokens=10, **generate_kwargs)
print(output)


# In[36]:


prompt =  "2 4 8 16 32"
output = tl_model.generate(prompt, max_new_tokens=8, **generate_kwargs)
print(output)


# In[ ]:


prompt =  "3 6 9 12 15"
output = tl_model.generate(prompt, max_new_tokens=15, **generate_kwargs)
print(output)


# # less than

# In[40]:


prompt =  "Bob is older than Steve. Bob is age 10 and Steve is age"
output = tl_model.generate(prompt, max_new_tokens=100, **generate_kwargs)
print(output)


# # Fibonacci

# In[45]:


prompt =  "0 1 1 2 3 5"
output = tl_model.generate(prompt, max_new_tokens=15, **generate_kwargs)
print(output)


# In[46]:


prompt =  "2 3 5 8"
output = tl_model.generate(prompt, max_new_tokens=15, **generate_kwargs)
print(output)


# In[ ]:





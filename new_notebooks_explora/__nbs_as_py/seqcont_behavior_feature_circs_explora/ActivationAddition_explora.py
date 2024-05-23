#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


# get the data and functions
get_ipython().system('git clone https://github.com/montemac/activation_additions.git')


# In[2]:


get_ipython().run_line_magic('cd', 'activation_additions')


# In[3]:


get_ipython().system('pip install -e .')


# In[3]:





# # load

# In[4]:


from typing import List

import torch
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import completion_utils, utils, hook_utils
from activation_additions.prompt_utils import (
    ActivationAddition,
    get_x_vector,
)


# In[ ]:


utils.enable_ipython_reload()

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda")


# # Generate vector

# ## love to hate

# In[5]:


activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="Love",
        prompt2="Hate",
        coeff=3,
        act_name=6,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="I hate you because you're",
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)


# In[6]:


love_rp, hate_rp = get_x_vector(prompt1="Love", prompt2="Hate",
                                coeff=5, act_name=6)


# ## first to one

# In[11]:


activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="first",
        prompt2="January",
        coeff=5,
        act_name=6,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="first second third",
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)


# In[12]:


activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="ranks",
        prompt2="months",
        coeff=5,
        act_name=6,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="first second third",
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)


# In[ ]:





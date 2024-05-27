#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


get_ipython().run_cell_magic('capture', '', 'try:\n    import google.colab # type: ignore\n    from google.colab import output\n    %pip install sae-lens==1.3.0 transformer-lens==1.17.0 circuitsvis==1.43.2\nexcept:\n    from IPython import get_ipython # type: ignore\n    ipython = get_ipython(); assert ipython is not None\n    ipython.run_line_magic("load_ext", "autoreload")\n    ipython.run_line_magic("autoreload", "2")\n')


# In[2]:


import torch
import os

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.lm_runner import language_model_sae_runner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # Load

# In[3]:


from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

# there are no pretrained SAEs on MLP layers, only on res stream output layers
hook_point = "blocks.8.hook_resid_pre"

# if the SAEs were stored with precomputed feature sparsities,
#  those will be return in a dictionary as well.
saes, sparsities = get_gpt2_res_jb_saes(hook_point)

print(saes.keys())


# In[4]:


from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader

sparse_autoencoder = saes[hook_point]
sparse_autoencoder.to(device)
sparse_autoencoder.cfg.device = device

loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)

# don't overwrite the sparse autoencoder with the loader's sae (newly initialized)
model, _, activation_store = loader.load_sae_training_group_session()

# TODO: We should have the session loader go straight to huggingface.


# In[ ]:





"""
Types of functions in this file:
- metrics
- reset states

"""

# import ast
# from collections import OrderedDict
# import re
# import sys
# import time
# from collections import defaultdict
# from enum import Enum
# from huggingface_hub import hf_hub_download

# import numpy as np
# import torch
# import torch.nn.functional as F
# import wandb

# from transformer_lens.HookedTransformer import HookedTransformer

# #https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/acdc_utils.py
# def kl_divergence(
#     logits: torch.Tensor,
#     base_model_logprobs: torch.Tensor,
#     mask_repeat_candidates: Optional[torch.Tensor] = None,
#     last_seq_element_only: bool = True,
#     base_model_probs_last_seq_element_only: bool = False,
#     return_one_element: bool = True,
# ) -> torch.Tensor:
#     # Note: we want base_model_probs_last_seq_element_only to remain False by default, because when the Docstring
#     # circuit uses this, it already takes the last position before passing it in.

#     if last_seq_element_only:
#         logits = logits[:, -1, :]

#     if base_model_probs_last_seq_element_only:
#         base_model_logprobs = base_model_logprobs[:, -1, :]

#     logprobs = F.log_softmax(logits, dim=-1)
#     kl_div = F.kl_div(logprobs, base_model_logprobs, log_target=True, reduction="none").sum(dim=-1)

#     if mask_repeat_candidates is not None:
#         assert kl_div.shape == mask_repeat_candidates.shape, (kl_div.shape, mask_repeat_candidates.shape)
#         answer = kl_div[mask_repeat_candidates]
#     elif not last_seq_element_only:
#         assert kl_div.ndim == 2, kl_div.shape
#         answer = kl_div.view(-1)
#     else:
#         answer = kl_div

#     if return_one_element:
#         return answer.mean()

#     return answer
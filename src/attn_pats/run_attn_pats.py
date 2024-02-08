"""
Runs visualize attention patterns and saves to json

Usage:
python run_attn_pats.py --model "gpt2-small" --task "numerals" --num_samps 300 
"""
import os
import pickle
import argparse
from transformer_lens import HookedTransformer

from viz_attn_pat import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-small")
    parser.add_argument("--task", choices=["numerals", "numwords", "months"], type=str, default="numerals")
    parser.add_argument("--num_samps", type=int, default=512)

    args = parser.parse_args()
    model_name = args.model
    task = args.task  # choose: numerals, numwords, months
    num_samps_per_ptype = args.num_samps #768 512

    ### Load Model ###
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )


    ### Load Datasets ###
    prompt_types = ['done', 'lost', 'names']

    # save_files = True
    # run_on_other_tasks = True
    prompts_list = []

    for i in prompt_types:
        # file_name = f'/content/seqcont_circ_expms/data/{task}/{task}_prompts_{i}.pkl'
        file_name = f'../../data/{task}/{task}_prompts_{i}.pkl'
        with open(file_name, 'rb') as file:
            filelist = pickle.load(file)

        print(filelist[0]['text'])
        prompts_list += filelist [:num_samps_per_ptype]
    prompts = [prompt['text'] for prompt in prompts_list]
    
    tokens = model.to_tokens(prompts, prepend_bos=True)
    # tokens = tokens.cuda() # Move the tokens to the GPU

    # get the cache to get attention patterns from
    original_logits, local_cache = model.run_with_cache(tokens) # Run the model and cache all activations

    ### Visualize ###
    layer = 1
    head_ind = 5
    viz_attn_pat(
        model,
        tokens,
        local_cache,
        layer, 
        head_ind,
        task,
        highlightLines = 'early',
        savePlotName = f'attnpat{layer}_{head_ind}_{task}'
    )

    layer = 4
    head_ind = 4
    viz_attn_pat(
        model,
        tokens,
        local_cache,
        layer, 
        head_ind,
        task,
        highlightLines = 'early',
        savePlotName = f'attnpat{layer}_{head_ind}_{task}'
    )

    layer = 7
    head_ind = 11
    viz_attn_pat(
        model,
        tokens,
        local_cache,
        layer, 
        head_ind,
        task,
        highlightLines = 'mid',
        savePlotName = f'attnpat{layer}_{head_ind}_{task}'
    )

    layer = 9
    head_ind = 1
    viz_attn_pat(
        model,
        tokens,
        local_cache,
        layer, 
        head_ind,
        task,
        highlightLines = 'late',
        savePlotName = f'attnpat{layer}_{head_ind}_{task}'
    )


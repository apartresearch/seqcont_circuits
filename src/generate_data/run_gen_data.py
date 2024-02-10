"""
Runs

Usage:
python run_gen_data.py --model "gpt2" 
--task "numerals" --num_samps 10
"""
import os
import pickle
import json
import argparse

import torch
from typing import Optional
import copy

import transformer_lens.utils as utils  # for test prompts
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)

from generate_data import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")  # 'gpt2' is small
    # parser.add_argument("--task", choices=["numerals", "numwords", "months"], type=str, default="numerals")
    # parser.add_argument("--num_samps", type=int, default=512)

    args = parser.parse_args()
    model_name = args.model 
    # task = args.task  # choose: numerals, numwords, months
    # num_samps_per_ptype = args.num_samps #768 512

    ### Load Model ###
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    
    #### Run gen dataset ####

    ### numwords, months- names
    from faker import Faker

    # Generate 100 unique first names
    fake = Faker()
    first_names = set()
    while len(first_names) < 500:
        first_name = fake.first_name()
        first_names.add(first_name)
    first_names = list(first_names)
    names = filter_to_single_token(model, first_names)

    # "Claire was born in February. John was born in March. Eve was born in April. Bob was born in”
    prompts_list = generate_prompts_list(0, 8, names[:100], 'born')

    # Replace the month names in the data
    prompts_list_months = replace_nw_seqtype(prompts_list, 'months')

    good_prompts, good_prompts_months, all_probs = get_good_prompts_nw_months(model, prompts_list, prompts_list_months)

    with open('nw_prompts_names.pkl', 'wb') as file:
        pickle.dump(good_prompts, file)
    with open('months_prompts_names.pkl', 'wb') as file:
        pickle.dump(good_prompts_months, file)

    ### numwords, months- lost
    # List of common, short words which are likely to be single tokens in GPT-2
    with open('common_words.json', 'r') as file:
        common_words = json.load(file)

    random_single_word_objects = [obj.capitalize() for obj in common_words]
    random_single_word_objects = filter_to_single_token(model, random_single_word_objects)
    
    prompts_list = generate_prompts_list(0, 8, random_single_word_objects, 'lost')

    prompts_list_months = replace_nw_seqtype(prompts_list, 'months')

    good_prompts, good_prompts_months, all_probs = get_good_prompts_nw_months(model, prompts_list, prompts_list_months)

    with open('nw_prompts_lost.pkl', 'wb') as file:
        pickle.dump(good_prompts, file)
    with open('months_prompts_lost.pkl', 'wb') as file:
        pickle.dump(good_prompts_months, file)

    ### numwords, months- done
    prompts_list = generate_prompts_list(0, 8, random_single_word_objects, 'done')
    prompts_list_months = replace_nw_seqtype(prompts_list, 'months')
    good_prompts, good_prompts_months, all_probs = get_good_prompts_nw_months(model, prompts_list, prompts_list_months)

    with open('nw_prompts_done.pkl', 'wb') as file:
        pickle.dump(good_prompts, file)
    with open('months_prompts_done.pkl', 'wb') as file:
        pickle.dump(good_prompts_months, file)

    ### numerals- names
    file_name = 'nw_prompts_names.pkl'
    with open(file_name, 'rb') as file:
        prompts_list = pickle.load(file)

    prompts_list = replace_nw_seqtype(prompts_list, 'numerals')
    good_prompts, all_probs = get_good_prompts_numerals(model, prompts_list)

    with open('digits_prompts_names.pkl', 'wb') as file:
        pickle.dump(good_prompts, file)

    ### numerals- lost
    file_name = 'nw_prompts_lost.pkl'
    with open(file_name, 'rb') as file:
        prompts_list = pickle.load(file)

    prompts_list = replace_nw_seqtype(prompts_list, 'numerals')
    good_prompts, all_probs = get_good_prompts_numerals(model, prompts_list)
    with open('digits_prompts_lost.pkl', 'wb') as file:
        pickle.dump(good_prompts, file)

    ### numerals- done
    file_name = 'nw_prompts_done.pkl'
    with open(file_name, 'rb') as file:
        prompts_list = pickle.load(file)

    prompts_list = replace_nw_seqtype(prompts_list, 'numerals')
    good_prompts, all_probs = get_good_prompts_numerals(model, prompts_list)
    with open('digits_prompts_done.pkl', 'wb') as file:
        pickle.dump(good_prompts, file)
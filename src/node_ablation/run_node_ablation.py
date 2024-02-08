"""
Runs iterative node ablation and saves circuit components to json

Usage:
python run_node_ablation.py --model "gpt2-small" --task "numerals" --num_samps 300 --threshold 20 --one_iter
"""
import os
import pickle
import json
import argparse

from dataset import Dataset
from generate_data import *
from metrics import *
from head_ablation_fns import *
from mlp_ablation_fns import *
from node_ablation_fns import *
from loop_node_ablation_fns import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-small")
    parser.add_argument("--task", choices=["numerals", "numwords", "months"], type=str, default="numerals")
    parser.add_argument("--num_samps", type=int, default=512)
    parser.add_argument("--threshold", type=int, default=20)
    parser.add_argument("--one_iter", action="store_true", default=False)

    args = parser.parse_args()
    model_name = args.model
    task = args.task  # choose: numerals, numwords, months
    num_samps_per_ptype = args.num_samps #768 512
    threshold = args.threshold
    one_iter = args.one_iter

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
        file_name = f'data/{task}/{task}_prompts_{i}.pkl'
        with open(file_name, 'rb') as file:
            filelist = pickle.load(file)

        print(filelist[0]['text'])
        prompts_list += filelist [:num_samps_per_ptype]

    pos_dict = {}
    for i in range(len(model.tokenizer.tokenize(prompts_list[0]['text']))):
        pos_dict['S'+str(i)] = i

    dataset = Dataset(prompts_list, pos_dict, model.tokenizer)

    # file_name = f'/content/seqcont_circ_expms/data/{task}/randDS_{task}.pkl'
    file_name = f'data/{task}/randDS_{task}.pkl'
    with open(file_name, 'rb') as file:
        prompts_list_2 = pickle.load(file)

    dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)

    #### Get orig score ####
    model.reset_hooks(including_permanent=True)
    ioi_logits_original = model(dataset.toks)
    orig_score = get_logit_diff(ioi_logits_original, dataset)
    # print(orig_score)

    ##############
    ### Node Ablation Iteration ###

    curr_circ_heads = []
    curr_circ_mlps = []
    prev_score = 100
    new_score = 0
    iter = 1
    all_comp_scores = []
    while prev_score != new_score:
        print('\nbackw prune, iter ', str(iter))
        old_circ_heads = curr_circ_heads.copy() # save old before finding new one
        old_circ_mlps = curr_circ_mlps.copy()
        curr_circ_heads, curr_circ_mlps, new_score, comp_scores = find_circuit_backw(model, dataset, dataset_2, curr_circ_heads, curr_circ_mlps, orig_score, threshold)
        if old_circ_heads == curr_circ_heads and old_circ_mlps == curr_circ_mlps:
            break
        all_comp_scores.append(comp_scores)
        print('\nfwd prune, iter ', str(iter))
        # track changes in circuit as for some reason it doesn't work with scores
        old_circ_heads = curr_circ_heads.copy()
        old_circ_mlps = curr_circ_mlps.copy()
        curr_circ_heads, curr_circ_mlps, new_score, comp_scores = find_circuit_forw(model, dataset, dataset_2, curr_circ_heads, curr_circ_mlps, orig_score, threshold)
        if old_circ_heads == curr_circ_heads and old_circ_mlps == curr_circ_mlps:
            break
        all_comp_scores.append(comp_scores)
        if one_iter:
            break
        iter += 1

    # save to JSON
    circuit_dict = {
        'heads': curr_circ_heads,
        'mlps': curr_circ_mlps,
    }

    circ_file_name = f'new_results/{task}_circuit_thres_{threshold}.json'
    directory = os.path.dirname(circ_file_name)
    if not os.path.exists(directory):
        os.makedirs('new_results', exist_ok=True)
    with open(circ_file_name, 'w') as json_file:
        json.dump(circuit_dict, json_file, indent=4)

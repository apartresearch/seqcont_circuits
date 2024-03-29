"""
Runs iterative edge ablation and saves circuit to json

Usage:
python run_edge_ablation.py --model "gpt2-small" --task "numerals" --num_samps 512 --threshold 0.8
"""
import os
import gc
import pickle
import json
import argparse

import torch as t
from torch import Tensor
from jaxtyping import Float

from transformer_lens import HookedTransformer

from edge_pruning_fns import *
from viz_circuits import *

from dataset import Dataset
from metrics import *

import sys
sys.path.append('../iter_node_pruning')
from head_ablation_fns import *
from mlp_ablation_fns import *
from node_ablation_fns import *
from loop_node_ablation_fns import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-small")
    parser.add_argument("--task", choices=["numerals", "numwords", "months"], type=str, default="numerals")
    parser.add_argument("--num_samps", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.8)

    args = parser.parse_args()
    model_name = args.model
    task = args.task  # choose: numerals, numwords, months
    num_samps_per_ptype = args.num_samps #768 512
    threshold = args.threshold

    save_files = True
    # run_on_other_tasks = True

    ### Load Model ###
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )

    ### Load Circuit ###
    #Load the saved heads_not_ablate and mlps_not_ablate json instead of passing as args (too large)

    circ_file_name = f'../../results/{task}_circuit_thres_{threshold}.json'
    with open(circ_file_name, 'r') as json_file:
        circuit_dict = json.load(json_file)
    heads_not_ablate, mlps_not_ablate = circuit_dict['heads'], circuit_dict['mlps']

    ### Load Datasets ###
    prompt_types = ['done', 'lost', 'names']

    prompts_list = []

    for i in prompt_types:
        file_name = f'../../data/{task}/{task}_prompts_{i}.pkl'
        with open(file_name, 'rb') as file:
            filelist = pickle.load(file)

        print(filelist[0]['text'])
        prompts_list += filelist [:num_samps_per_ptype]

    pos_dict = {}
    for i in range(len(model.tokenizer.tokenize(prompts_list[0]['text']))):
        pos_dict['S'+str(i)] = i

    dataset_1 = Dataset(prompts_list, pos_dict, model.tokenizer)

    file_name = f'../../data/{task}/randDS_{task}.pkl'
    with open(file_name, 'rb') as file:
        prompts_list_2 = pickle.load(file)

    dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)

    #### Get orig score ####
    model.reset_hooks(including_permanent=True)
    logits_original = model(dataset_1.toks)
    orig_score = get_logit_diff(logits_original, dataset_1)
    # print(orig_score)

    def logit_diff_perc(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float = orig_score,
        dataset_1: Dataset = dataset_1,
    ) -> float:
        patched_logit_diff = get_logit_diff(logits, dataset_1)
        return (patched_logit_diff / clean_logit_diff)

    del(logits_original)
    t.cuda.empty_cache()
    gc.collect()

    ##############
    ### Edge Ablation Iteration ###

    ### get circuit

    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook
    abl_model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

    new_logits = model(dataset_1.toks)
    new_score = get_logit_diff(new_logits, dataset_1)
    circ_score = (100 * new_score / orig_score).item()
    print(f"(cand circuit / full) %: {circ_score:.4f}")
    del(new_logits)

    ### head to head
    qkv_to_HH = {} # qkv to dict

    for head_type in ["q", "k", "v"]:
        head_to_head_results = {}
        for head in heads_not_ablate:
            print(head_type, head)
            model.reset_hooks()
            model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

            result = circ_path_patch_head_to_heads(
                circuit = heads_not_ablate,
                receiver_heads = [head],
                receiver_input = head_type,
                model = model,
                patching_metric = logit_diff_perc,
                new_dataset = dataset_2,
                orig_dataset = dataset_1
            )
            head_to_head_results[head] = result
        qkv_to_HH[head_type] = head_to_head_results

    head_to_head_adjList = {}
    for head_type in ["q", "k", "v"]:
        for head in heads_not_ablate:
            result = qkv_to_HH[head_type][head]
            filtered_indices = (result < threshold) & (result != 0.0)
            rows, cols = filtered_indices.nonzero(as_tuple=True)
            sender_nodes = list(zip(rows.tolist(), cols.tolist()))
            head_with_type = head + (head_type,)
            head_to_head_adjList[head_with_type] = sender_nodes

    ### mlp to mlp
    mlp_to_mlp_results = {}

    for layer in reversed(mlps_not_ablate):
        print(layer)
        model.reset_hooks()
        model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)
        result = circ_path_patch_MLPs_to_MLPs(
            mlp_circuit = mlps_not_ablate,
            receiver_layers = [layer],
            model = model,
            patching_metric = logit_diff_perc,
            new_dataset = dataset_2,
            orig_dataset = dataset_1
        )
        mlp_to_mlp_results[layer] = result

    mlp_to_mlp_adjList = {}
    for mlp in mlps_not_ablate:
        result = mlp_to_mlp_results[mlp]
        filtered_indices = (result < threshold) & (result != 0.0)
        filtered_indices = filtered_indices.nonzero(as_tuple=True)[0]
        mlp_to_mlp_adjList[mlp] = filtered_indices.tolist()

    ### head to mlp
    head_to_mlp_results = {}

    for layer in reversed(mlps_not_ablate):
        print(layer)
        model.reset_hooks()
        model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)
        result = circ_path_patch_head_to_mlp(
            circuit = heads_not_ablate,
            receiver_layers = [layer],
            model = model,
            patching_metric = logit_diff_perc,
            new_dataset = dataset_2,
            orig_dataset = dataset_1
        )
        head_to_mlp_results[layer] = result

    head_to_mlp_adjList = {}
    for layer in mlps_not_ablate:
        result = head_to_mlp_results[layer]
        filtered_indices = (result < threshold) & (result != 0.0)
        rows, cols = filtered_indices.nonzero(as_tuple=True)
        sender_nodes = list(zip(rows.tolist(), cols.tolist()))
        head_to_mlp_adjList[layer] = sender_nodes

    ### mlp to head
    qkv_mlp_to_HH = {} # qkv to dict

    for head_type in ["q", "k", "v"]:
        mlp_to_head_results = {}
        for head in heads_not_ablate:
            print(head_type, head)
            model.reset_hooks()
            model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

            result = circ_path_patch_mlp_to_head(
                mlp_circuit = mlps_not_ablate,
                receiver_heads = [head],
                receiver_input = head_type,
                model = model,
                patching_metric = logit_diff_perc,
                new_dataset = dataset_2,
                orig_dataset = dataset_1
            )
            mlp_to_head_results[head] = result
        qkv_mlp_to_HH[head_type] = mlp_to_head_results

    mlp_to_head_adjList = {}
    for head_type in ["q", "k", "v"]:
        for head in heads_not_ablate:
            result = qkv_mlp_to_HH[head_type][head]
            filtered_indices = (result < threshold) & (result != 0.0)
            filtered_indices = filtered_indices.nonzero(as_tuple=True)[0]
            head_with_type = head + (head_type,)
            mlp_to_head_adjList[head_with_type] = filtered_indices.tolist()

    ### Save graph files to free up memory
    if save_files:
        with open(task + "_head_to_head_results.pkl", "wb") as file:
            pickle.dump(head_to_head_results, file)

        with open(task + "_mlp_to_mlp_results.pkl", "wb") as file:
            pickle.dump(mlp_to_mlp_results, file)

        with open(task + "_head_to_mlp_results.pkl", "wb") as file:
            pickle.dump(head_to_mlp_results, file)

        with open(task + "_mlp_to_head_results.pkl", "wb") as file:
            pickle.dump(mlp_to_head_results, file)

    del(head_to_head_results)
    del(mlp_to_mlp_results)
    del(head_to_mlp_results)
    del(mlp_to_head_results)

    ### Run Iter Edge Pruning- resid post
    ## head to resid
    model.reset_hooks()
    model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

    path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(heads_not_ablate, model, logit_diff_perc,
                                                                new_dataset = dataset_2, orig_dataset = dataset_1)

    heads_to_resid = {}
    result = path_patch_head_to_final_resid_post
    filtered_indices = (result < threshold) & (result != 0.0)
    rows, cols = filtered_indices.nonzero(as_tuple=True)
    heads_to_resid['resid'] = list(zip(rows.tolist(), cols.tolist()))

    ## mlp to resid
    model.reset_hooks()
    model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)

    path_patch_mlp_to_final_resid_post = get_path_patch_mlp_to_final_resid_post(mlps_not_ablate, model, logit_diff_perc,
                                                                    new_dataset = dataset_2, orig_dataset = dataset_1)

    mlps_to_resid = {}
    result = path_patch_mlp_to_final_resid_post
    filtered_indices = (result < threshold) & (result != 0.0)
    filtered_indices = filtered_indices.nonzero(as_tuple=True)[0]
    mlps_to_resid['resid'] = filtered_indices.tolist()

    ### Filter out nodes with no ingoing edges
    head_to_head_adjList = {node: neighbors for node, neighbors in head_to_head_adjList.items() if neighbors}
    mlp_to_head_adjList = {node: neighbors for node, neighbors in mlp_to_head_adjList.items() if neighbors}

    ### Save rest of graph files
    if save_files:
        # graphs
        with open(task + "_head_to_head_adjList.pkl", "wb") as file:
            pickle.dump(head_to_head_adjList, file)

        with open(task + "_mlp_to_mlp_adjList.pkl", "wb") as file:
            pickle.dump(mlp_to_mlp_adjList, file)

        with open(task + "_head_to_mlp_adjList.pkl", "wb") as file:
            pickle.dump(head_to_mlp_adjList, file)

        with open(task + "_mlp_to_head_adjList.pkl", "wb") as file:
            pickle.dump(mlp_to_head_adjList, file)

        with open(task + "_heads_to_resid.pkl", "wb") as file:
            pickle.dump(heads_to_resid, file)

        with open(task + "_mlps_to_resid.pkl", "wb") as file:
            pickle.dump(mlps_to_resid, file)

        # score results
        with open(task + "_heads_to_resid_results.pkl", "wb") as file:
            pickle.dump(path_patch_head_to_final_resid_post, file)

        with open(task + "_mlps_to_resid_results.pkl", "wb") as file:
            pickle.dump(path_patch_mlp_to_final_resid_post, file)

    ####################
    ### Circuit graph plot ###
            
    plot_graph_adjacency_qkv(head_to_head_adjList, mlp_to_mlp_adjList, head_to_mlp_adjList,
                         mlp_to_head_adjList, heads_to_resid, mlps_to_resid, filename="qkv")
    
    plot_graph_adjacency(head_to_head_adjList, mlp_to_mlp_adjList, head_to_mlp_adjList,
                         mlp_to_head_adjList, heads_to_resid, mlps_to_resid, filename="no qkv")

    ### save to JSON
    # circuit_dict = {
    #     'heads': curr_circ_heads,
    #     'mlps': curr_circ_mlps,
    # }

    # circ_file_name = f'new_results/{task}_circuit_thres_{threshold}.json'
    # directory = os.path.dirname(circ_file_name)
    # if not os.path.exists(directory):
    #     os.makedirs('new_results', exist_ok=True)
    # with open(circ_file_name, 'w') as json_file:
    #     json.dump(circuit_dict, json_file, indent=4)

"""
Runs logit lens on MLP output layers (these are same as hidden state outputs, as MLP + layernorm is end of hidden state block)

Usage:
python run_logit_lens.py --model "gpt2" --task "numerals" --num_samps 10
"""
import os
import pickle
import json
import argparse

from logit_lens_fns import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")  # 'gpt2' is small
    parser.add_argument("--task", choices=["numerals", "numwords", "months"], type=str, default="numerals")
    parser.add_argument("--num_samps", type=int, default=512)

    args = parser.parse_args()
    model_name = args.model 
    task = args.task  # choose: numerals, numwords, months
    num_samps_per_ptype = args.num_samps #768 512

    ### Load Model ###
    device = 'cuda:0' if t.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=t.float16).to(device)

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

    #### Run logit lens on dataset ####
    print(1)


    # save to JSON
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

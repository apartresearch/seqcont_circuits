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

    if task == "numerals":
        print(1)
    elif task == "numwords":
        num_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                 "eleven", "twelve"]
    elif task == "months":
        num_words = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    anomolies = []
    num_corr = 0
    for pd in prompts_list:
        test_text = pd['text']
        # layer_logits = get_logits(test_text)
        # tok_logit_lens = get_decoded_indiv_toks(layer_logits)
        layer_logits = get_logits(model, tokenizer, device, test_text)
        tok_logit_lens = get_decoded_indiv_toks(tokenizer, layer_logits)

        """
        Check if the 8th layer's predicted token is the sequence member just "one before"
        the correct next sequence member output found in the ninth layer

        Use `try` because when indexing, the output may not be a seq member of the right type!
        """
        try:
            if task == "numerals":
                a = tok_logit_lens[9][0].replace(' ', '')
                b= tok_logit_lens[10][0].replace(' ', '')
            else:   
                a = num_words.index(tok_logit_lens[9][0].replace(' ', ''))
                b= num_words.index(tok_logit_lens[10][0].replace(' ', ''))
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

    print(num_corr)

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

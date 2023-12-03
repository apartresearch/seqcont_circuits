from torch import Tensor
from jaxtyping import Float, Int
from typing import Any, Optional, Tuple, Union, List, Dict

from dataset import Dataset
import circuit_extraction as circuit_extraction

# class ModelScores:
#     def __init__(self, model, dataset):
#         self.model = model
#         self.dataset = dataset

#         model.reset_hooks(including_permanent=True)
#         self.logits_original, self.cache = model.run_with_cache(dataset.toks)
#         self.orig_score = logits_to_ave_logit_diff(self.logits_original, dataset)

# may add fns below as methods to this class if deemed neater due to those fns being specific to (model, dataset)

def logits_to_ave_logit_diff(logits: Float[Tensor, "batch seq d_vocab"], dataset: Dataset, per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''

    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    corr_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.corr_tokenIDs]
    incorr_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.incorr_tokenIDs]
    # Find logit difference
    answer_logit_diff = corr_logits - incorr_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def mean_ablate_by_lst(
        lst, model, dataset: Dataset, dataset_2: Dataset, orig_score: float, 
        CIRCUIT: Dict[str, List[Tuple[int, int]]]=None, SEQ_POS_TO_KEEP: Dict[str, str]=None, print_output=True):
    if CIRCUIT == None:
        CIRCUIT = {
            "number mover": lst,
            # "number mover 4": lst,
            "number mover 3": lst,
            "number mover 2": lst,
            "number mover 1": lst,
        }

    if SEQ_POS_TO_KEEP == None:
        SEQ_POS_TO_KEEP = {
            "number mover": "end",
            # "number mover 4": "S4",
            "number mover 3": "S3",
            "number mover 2": "S2",
            "number mover 1": "S1",
        }

    model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

    # ioi_logits_original, ioi_cache = model.run_with_cache(dataset.toks)

    model = circuit_extraction.add_mean_ablation_hook(model, means_dataset=dataset_2, circuit=CIRCUIT, seq_pos_to_keep=SEQ_POS_TO_KEEP)
    ioi_logits_minimal = model(dataset.toks)

    # orig_score = logits_to_ave_logit_diff(ioi_logits_original, dataset)
    new_score = logits_to_ave_logit_diff(ioi_logits_minimal, dataset)
    if print_output:
        # print(f"Average logit difference (IOI dataset, using entire model): {orig_score:.4f}")
        # print(f"Average logit difference (IOI dataset, only using circuit): {new_score:.4f}")
        print(f"Average logit difference (circuit / full) %: {100 * new_score / orig_score:.4f}")
    # return new_score
    return 100 * new_score / orig_score

def find_circuit_forw(model, curr_circuit: Optional[List] = None, orig_score: float = 100, threshold: float = 10) -> Tuple:
    # threshold is T, a %. if performance is less than T%, allow its removal
    if curr_circuit == []:
        # Start with full circuit
        curr_circuit = [(layer, head) for layer in range(12) for head in range(12)]

    for layer in range(0, 12):
        for head in range(12):
            if (layer, head) not in curr_circuit:
                continue

            # Copying the curr_circuit so we can iterate over one and modify the other
            copy_circuit = curr_circuit.copy()

            # Temporarily removing the current tuple from the copied circuit
            copy_circuit.remove((layer, head))

            new_score = mean_ablate_by_lst(copy_circuit, model, orig_score, print_output=False).item()

            # print((layer,head), new_score)
            # If the result is less than the threshold, remove the tuple from the original list
            if (100 - new_score) < threshold:
                curr_circuit.remove((layer, head))

                print("\nRemoved:", (layer, head))
                print(new_score)

    return curr_circuit, new_score
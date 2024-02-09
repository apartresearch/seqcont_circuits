import torch as t
from torch import Tensor
from jaxtyping import Float
from typing import List, Optional, Callable, Tuple
from functools import partial

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, ActivationCache

from dataset import Dataset

def patch_head_vectors(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int, int],
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    '''
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector

def patch_head_input(
    orig_activation: Float[Tensor, "batch pos head_idx d_head"],
    hook: HookPoint,
    patched_cache: ActivationCache,
    head_list: List[Tuple[int, int]],
) -> Float[Tensor, "batch pos head_idx d_head"]:
    '''
    '''
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
    return orig_activation

###
def patch_mlp_vectors(
    orig_MLP_vector: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    layer_to_patch: int,
) -> Float[Tensor, "batch pos d_model"]:
    '''
    '''
    if layer_to_patch == hook.layer():
        orig_MLP_vector[:, :, :] = new_cache[hook.name][:, :, :]
    return orig_MLP_vector

def patch_mlp_input(
    orig_activation: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    patched_cache: ActivationCache,
    layer_list: List[int],
) -> Float[Tensor, "batch pos head_idx d_head"]:
    '''
    '''
    if hook.layer() in layer_list:
        orig_activation[:, :, :] = patched_cache[hook.name][:, :, :]
    return orig_activation

###############
def circ_path_patch_head_to_heads(
    circuit: List[Tuple[int, int]],
    receiver_heads: List[Tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: Dataset,
    orig_dataset: Dataset,
    new_cache: Optional[ActivationCache] = None,
    orig_cache: Optional[ActivationCache] = None,
) -> Float[Tensor, "layer head"]:
    '''
    Returns:
        tensor of scores of shape (max(receiver_layers), model.cfg.n_heads)
    '''
    assert receiver_input in ("k", "q", "v", "z")
    receiver_layers = set(next(zip(*receiver_heads)))  # a set of all layers of receiver heads
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    results = t.zeros(max(receiver_layers), model.cfg.n_heads, device="cuda", dtype=t.float32)

    ### 1. Get activations ###
    z_name_filter = lambda name: name.endswith("z")
    if new_cache is None:
        _, new_cache = model.run_with_cache(
            new_dataset.toks,
            names_filter=z_name_filter,
            return_type=None
        )
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=z_name_filter,
            return_type=None
        )

    senders = [tup for tup in circuit if tup[0] < receiver_heads[0][0]]  # senders are in layer before receivers

    for (sender_layer, sender_head) in senders:
        ### 2. Frozen Clean Run with sender node patched from Corrupted Run ###
        hook_fn = partial(
            patch_head_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn, level=1)

        _, patched_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=receiver_hook_names_filter,
            return_type=None
        )
        # model.reset_hooks(including_permanent=True)
        assert set(patched_cache.keys()) == set(receiver_hook_names)

        ### 3. Clean Run, with patched in received node from step 2 ###
        hook_fn = partial(
            patch_head_input,
            patched_cache=patched_cache,
            head_list=receiver_heads,
        )
        patched_logits = model.run_with_hooks(
            orig_dataset.toks,
            fwd_hooks = [(receiver_hook_names_filter, hook_fn)],
            return_type="logits"
        )

        results[sender_layer, sender_head] = patching_metric(patched_logits)

    return results

###############
def circ_path_patch_MLPs_to_MLPs(
    mlp_circuit: List[int],
    receiver_layers: List[int],
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: Dataset,
    orig_dataset: Dataset,
    new_cache: Optional[ActivationCache] = None,
    orig_cache: Optional[ActivationCache] = None,
) -> Float[Tensor, "layer head"]:
    '''
    Returns:
        tensor of scores of shape (max(receiver_layers), model.cfg.n_heads)
    '''
    # model.reset_hooks()
    receiver_hook_names = [utils.get_act_name('mlp_out', layer) for layer in receiver_layers]  # modify for mlp_out
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    results = t.zeros(max(receiver_layers), device="cuda", dtype=t.float32)

    ### 1. Get activations ###
    z_name_filter = lambda name: name.endswith("mlp_out")

    if new_cache is None:
        _, new_cache = model.run_with_cache(
            new_dataset.toks,
            names_filter=z_name_filter,
            return_type=None
        )
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=z_name_filter,
            return_type=None
        )

    sender_mlp_list = [L for L in mlp_circuit if L < max(receiver_layers)]
    for (sender_layer) in sender_mlp_list:
        ### 2. Frozen Clean Run with sender node patched from Corrupted Run ###
        hook_fn = partial(
            patch_mlp_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            layer_to_patch = sender_layer
        )

        model.add_hook(z_name_filter, hook_fn, level=1)

        _, patched_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=receiver_hook_names_filter,
            return_type=None
        )
        # model.reset_hooks(including_permanent=True)
        # assert set(patched_cache.keys()) == set(receiver_hook_names)

        ### 3. Clean Run with patched receiver node from step 2 ###
        hook_fn = partial(
            patch_mlp_input,
            patched_cache=patched_cache,
            layer_list=receiver_layers,
        )
        patched_logits = model.run_with_hooks(
            orig_dataset.toks,
            fwd_hooks = [(receiver_hook_names_filter, hook_fn)],
            return_type="logits"
        )

        results[sender_layer] = patching_metric(patched_logits)

    # the result is which sender layers affect ALL the inputted nodes. this is why we just
    # want to pass one node at a time- to see which layers affect just IT.
    # if we want a 'group of nodes under a common type', we'd pass a set of nodes
    return results

###############
def circ_path_patch_head_to_mlp(
    circuit: List[Tuple[int, int]],
    receiver_layers: List[int],
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: Dataset,
    orig_dataset: Dataset,
    new_cache: Optional[ActivationCache] = None,
    orig_cache: Optional[ActivationCache] = None,
) -> Float[Tensor, "layer head"]:
    '''
    Returns:
        tensor of scores of shape (max(receiver_layers), model.cfg.n_heads)
    '''
    # model.reset_hooks()

    receiver_hook_names = [utils.get_act_name('mlp_out', layer) for layer in receiver_layers]  
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    results = t.zeros(max(receiver_layers), model.cfg.n_heads, device="cuda", dtype=t.float32)

    ### 1. Get activations ###
    z_name_filter = lambda name: name.endswith(("z", "mlpout"))  # gets same value as just z

    if new_cache is None:
        _, new_cache = model.run_with_cache(
            new_dataset.toks,
            names_filter=z_name_filter,
            return_type=None
        )
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=z_name_filter,
            return_type=None
        )

    senders = [tup for tup in circuit if tup[0] < receiver_layers[0]]
    for (sender_layer, sender_head) in senders:
        ### 2. Frozen Clean Run with sender node patched from Corrupted Run ###
        hook_fn = partial(
            patch_head_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )

        model.add_hook(z_name_filter, hook_fn, level=1)

        _, patched_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=receiver_hook_names_filter,
            return_type=None
        )
        # model.reset_hooks(including_permanent=True)
        # assert set(patched_cache.keys()) == set(receiver_hook_names)

        ### 3. Clean Run with patched receiver node from step 2 ###

        hook_fn = partial(
            patch_mlp_input,
            patched_cache=patched_cache,
            layer_list=receiver_layers,
        )
        patched_logits = model.run_with_hooks(
            orig_dataset.toks,
            fwd_hooks = [(receiver_hook_names_filter, hook_fn)],
            return_type="logits"
        )

        results[sender_layer, sender_head] = patching_metric(patched_logits)

    return results

###############
def circ_path_patch_mlp_to_head(
    mlp_circuit: List[int],
    receiver_heads: List[Tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: Dataset,
    orig_dataset: Dataset,
    new_cache: Optional[ActivationCache] = None,
    orig_cache: Optional[ActivationCache] = None,
) -> Float[Tensor, "layer head"]:
    '''
    Returns:
        tensor of scores of shape (max(receiver_layers), model.cfg.n_heads)
    '''
    # model.reset_hooks() # doesn't make diff if comment out or not

    assert receiver_input in ("k", "q", "v", "z")  # we can run get_path_patch_head_to_heads() 3 times for k, q, v!
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    results = t.zeros(max(receiver_layers), device="cuda", dtype=t.float32)

    ### 1. Get activations ###
    z_name_filter = lambda name: name.endswith(("z", "mlpout"))  # gets same value as just mlp out

    if new_cache is None:
        _, new_cache = model.run_with_cache(
            new_dataset.toks,
            names_filter=z_name_filter,
            return_type=None
        )
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=z_name_filter,
            return_type=None
        )

    sender_mlp_list = [L for L in mlp_circuit if L < receiver_heads[0][0]]
    for (sender_layer) in sender_mlp_list:
        ### 2. Frozen Clean Run with sender node patched from Corrupted Run ###
        hook_fn = partial(
            patch_mlp_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            layer_to_patch = sender_layer
        )

        model.add_hook(z_name_filter, hook_fn, level=1)

        _, patched_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=receiver_hook_names_filter,
            return_type=None
        )
        # model.reset_hooks(including_permanent=True)
        assert set(patched_cache.keys()) == set(receiver_hook_names)

        ### 3. Clean Run with patched receiver node from step 2 ###
        hook_fn = partial(
            patch_head_input,
            patched_cache=patched_cache,
            head_list=receiver_heads,
        )
        patched_logits = model.run_with_hooks(
            orig_dataset.toks,
            fwd_hooks = [(receiver_hook_names_filter, hook_fn)],
            return_type="logits"
        )

        results[sender_layer] = patching_metric(patched_logits)

    return results

###############
def get_path_patch_head_to_final_resid_post(
    circuit: List[Tuple[int, int]],
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: Dataset,
    orig_dataset: Dataset,
) -> Float[Tensor, "layer head"]:
    '''
    Returns:

    '''
    # model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device="cuda", dtype=t.float32)

    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_post_name_filter = lambda name: name == resid_post_hook_name


    ### 1. Get activations ###
    z_name_filter = lambda name: name.endswith("z")

    _, new_cache = model.run_with_cache(
        new_dataset.toks,
        names_filter=z_name_filter,
        return_type=None
    )

    _, orig_cache = model.run_with_cache(
        orig_dataset.toks,
        names_filter=z_name_filter,
        return_type=None
    )

    for (sender_layer, sender_head) in circuit:
        ### 2. Frozen Clean Run with sender node patched from Corrupted Run ###
        hook_fn = partial(
            patch_head_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn)

        _, patched_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=resid_post_name_filter,
            return_type=None
        )

        assert set(patched_cache.keys()) == {resid_post_hook_name}

        ### 3. Clean Run with patched receiver node from step 2 ###
        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))

        results[sender_layer, sender_head] = patching_metric(patched_logits)

    return results

###############
def get_path_patch_mlp_to_final_resid_post(
    mlp_circuit: List[int],
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: Dataset,
    orig_dataset: Dataset,
) -> Float[Tensor, "layer head"]:
    '''
    '''
    # model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, device="cuda", dtype=t.float32) #model.cfg.n_heads,

    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_post_name_filter = lambda name: name == resid_post_hook_name


    ### 1. Get activations ###
    z_name_filter = lambda name: name.endswith(("z", "mlp_out"))

    _, new_cache = model.run_with_cache(
        new_dataset.toks,
        names_filter=z_name_filter,
        return_type=None
    )

    _, orig_cache = model.run_with_cache(
        orig_dataset.toks,
        names_filter=z_name_filter,
        return_type=None
    )

    for sender_layer in mlp_circuit:
        ### 2. Frozen Clean Run with sender node patched from Corrupted Run ###
        hook_fn = partial(
            patch_mlp_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            layer_to_patch = sender_layer
        )
        model.add_hook(z_name_filter, hook_fn)

        _, patched_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=resid_post_name_filter,
            return_type=None
        )

        assert set(patched_cache.keys()) == {resid_post_hook_name}

        ### 3. Clean Run with patched receiver node from step 2 ###

        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))

        results[sender_layer] = patching_metric(patched_logits)

    return results
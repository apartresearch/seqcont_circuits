from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import torch.nn.functional as F

# consider using a wrapper instead of passing model and tokenzier in every time

def actvs_to_logits(model, hidden_states):
    """
    outputs.hidden_states is a tuple for every layer
    each tuple member is an actvs tensor of size (batch_size, seq_len, d_model)
    loop thru tuple to get actv for each layer
    """
    layer_logits_list = []  # logits for each layer hidden state output actvs
    for i, h in enumerate(hidden_states):
        h_last_tok = h[:, -1, :] 
        if i == len(hidden_states) - 1:
            h_last_tok = model.transformer.ln_f(h_last_tok)  # apply layer norm as not in last 
        logits = t.einsum('ab,cb->ac', model.lm_head.weight, h_last_tok)
        layer_logits_list.append(logits)
    return layer_logits_list

def get_logits(model, tokenizer, device, input_text):
    token_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = model(token_ids, output_hidden_states=True)
    # logits = actvs_to_logits(outputs.hidden_states)
    logits = actvs_to_logits(model, outputs.hidden_states)
    logits = t.stack(logits).squeeze(-1)
    return logits

def get_decoded_indiv_toks(tokenizer, layer_logits, k=10):
    """
    i is the layer (from before to last). 
    layer_logits[i] are the scores for each token in vocab dim for the ith unembedded layer
    j is the top 5
    """
    output_list = []
    for i in range(len(layer_logits)):
        top_5_at_layer = []
        sorted_token_ids = F.softmax(layer_logits[i],dim=-1).argsort(descending=True)
        for j in range(5):  # loop to separate them in a list, rather than concat into one str
            top_5_at_layer.append(tokenizer.decode(sorted_token_ids[j]))
        output_list.append( top_5_at_layer )
    return output_list
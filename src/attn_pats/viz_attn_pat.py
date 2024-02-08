import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def get_ind(token_list, token1, token2, printInd=False):
    # Find the indices of the tokens in the tokenized sentence
    try:
        query_ind = token_list.index(token1)
        key_ind = token_list.index(token2)
    except ValueError as e:
        print(f"Token not found: {e}")
    else:
        if printInd:
            print(f"The index of '{token1}' is {query_ind}")
            print(f"The index of '{token2}' is {key_ind}")
    return query_ind, key_ind

def viz_attn_pat(
    model,
    tokens,
    local_cache,
    layer, 
    head_index,
    task = 'numerals', # choose: numerals, numwords, months
    highlightLines = '',  # early, mid, late, ''
    savePlotName = ''
):
    patterns = local_cache["attn", layer][:, head_index].mean(dim=0)
    patterns_np = patterns.cpu().numpy() 

    local_tokens = tokens[0]
    str_tokens = model.to_str_tokens(local_tokens)
    str_tokens[0] = '<PAD>' # Rename the first token string as '<END>'

    # Create a mask for the cells above the diagonal
    mask = np.triu(np.ones_like(patterns_np, dtype=bool), k=1)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        patterns_np,
        xticklabels=str_tokens,
        yticklabels=str_tokens,
        cmap = 'inferno',
        annot=False,
        fmt='.2f',
        linewidths=0.1,  # Set linewidth to create grid lines between cells
        linecolor='white', 
        mask=mask
    )

    ax.set_xlabel('Key', fontsize=16, fontweight='bold', labelpad=20)
    ax.set_ylabel('Query', fontsize=16, fontweight='bold', labelpad=20)

    # choose tokens to display 
    if task == 'numerals':
        disp_toks = [" 4", " 3", " 2", " 1"]
    elif task == 'numwords':
        disp_toks = [" four", " three", " two", " one"]
    elif task == 'months':
        disp_toks = [" April", " March", " February", " January"]
        # disp_toks = [" Apr", " Mar", " Feb", " Jan"]

    if highlightLines != '':
        if highlightLines == 'early':
            token_pairs_highL = [(disp_toks[-2], disp_toks[-1]), (disp_toks[-3], disp_toks[-2]), (disp_toks[-4], disp_toks[-3])]
            # token_pairs_highL = [(" 2", " 1"), (" 3", " 2"), (" 4", " 3")]
        elif highlightLines == 'mid':
            last_token = str_tokens[-1]
            token_pairs_highL = [(last_token, disp_toks[-2]), (last_token, disp_toks[-3]), (last_token, disp_toks[-4])]
            # token_pairs_highL = [(last_token, ' 2'), (last_token, ' 3'), (last_token, ' 4')]
        elif highlightLines == 'late': 
            last_token = str_tokens[-1]
            token_pairs_highL = [(last_token, disp_toks[-4])]
            # token_pairs_highL = [(last_token, ' 4')]

        for qk_toks in token_pairs_highL:
            qInd, kInd = get_ind(str_tokens, qk_toks[0], qk_toks[1])
            if highlightLines != 'early':  # do this if last token has multiple repeats
                qInd = len(str_tokens) - 1  # or else it'd get the first repeat
            plt.plot([0, kInd+1], [qInd, qInd], color='#7FFF00', linewidth=5)  # top of highL row
            plt.plot([kInd+1, kInd+1], [qInd, len(str_tokens)], color='blue', linewidth=5)  # right of highL col

            yticklabels = ax.get_yticklabels()
            yticklabels[qInd].set_color('green')
            yticklabels[qInd].set_fontweight('bold')
            yticklabels[qInd].set_fontsize(14)
            ax.set_yticklabels(yticklabels)
            xticklabels = ax.get_xticklabels()
            xticklabels[kInd].set_color('blue')
            xticklabels[kInd].set_fontweight('bold')
            xticklabels[kInd].set_fontsize(14)
            ax.set_xticklabels(xticklabels)

        # offset pattern for prev token heads
        # if highlightLines == 'mid':
            # for i in range(0, len(str_tokens)-1):
                #     rect = patches.Rectangle((i, i+4), 1, 1, linewidth=4.5, edgecolor='#3CB371', facecolor='none')
                #     ax.add_patch(rect)

    if savePlotName != '':
        # file_name = f'new_results/savePlotName + '.png'
        # directory = os.path.dirname(file_name)
        # if not os.path.exists(directory):
        #     os.makedirs('new_results', exist_ok=True)
        plt.savefig(savePlotName + '.png', bbox_inches='tight')
        
    plt.show()
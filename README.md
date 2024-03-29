# Understanding Sequence Continuation in Transformers through Circuit Analysis 

## Overview

This repository hosts the code and resources used in our research on interpreting transformer models, specifically focusing on the GPT-2 architecture. Our work extends existing efforts in reverse engineering transformer models into human-readable circuits. We delve into the interpretability of these models by analyzing and comparing circuits involved in sequence continuation tasks, including sequences of Arabic numerals, number words, and months.

### Key Findings

- **Circuit Interpretability Analysis**: We successfully identified a crucial sub-circuit within GPT-2 responsible for detecting sequence members and predicting the next member in a sequence.
- **Shared Circuit Subgraphs**: Our research reveals that semantically related sequences utilize shared circuit subgraphs with analogous roles.
- **Model Behavior Predictions and Error Identification**: Documenting these shared computational structures aids in better predictions of model behavior, identifying potential errors, and formulating safer editing procedures.
- **Towards Robust, Aligned, and Interpretable Language Models**: Our findings contribute to the broader goal of creating language models that are not only powerful but also robust, aligned with human values, and interpretable.


#### To get started with our project, follow these steps:

Clone the Repository: 

`` git clone [repository URL] ``

Install Dependencies:

`` pip install -r requirements.txt ``

**Explore the Notebooks:**

Navigate to the ``notebooks`` directory and open the Colab notebooks to see detailed analyses and visualizations.

### Running Experiments

After navigating to the `src/iter_node_pruning` folder, use this command to run node ablation experiments. Lower `--num_samps` if one encounters GPU out-of-memory issues. An A100 is recommended. Change the task and other input parameters to run a different experiment.

```bash
python run_node_ablation.py --model "gpt2-small" --task "numerals" --num_samps 512 --threshold 20 --one_iter
```

Similarly, the other files can be run by naviating to their respective sub-folder in `src`. These other commands include:

```bash
python run_gen_data.py --model "gpt2-small" 

python run_edge_ablation.py --model "gpt2-small" --task "numerals" --num_samps 512 --threshold 0.8

python run_attn_pats.py --model "gpt2-small" --task "numerals" --num_samps 128 

# If an issue such as `RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'`, it could be due to the GPU not being powerful enough.
python run_logit_lens.py --model "gpt2" --task "numerals" --num_samps 512
```


# Understanding Transformers through Circuit Analysis 
##### by Michael Lan and Fazl Barez

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

Explore the Notebooks:

Navigate to the ``notebooks`` directory and open the Jupyter notebooks to see detailed analyses and visualizations.

#### Running Experiments

Use this command to run node ablation experiments. Lower `--num_samps` if the GPU out-of-memory issues.

```bash
python run_node_ablation.py --model "gpt2-small" --task "numerals" --num_samps 300 --threshold 20 --one_iter
```

#### Citation
If you find our work useful, please consider citing our paper:

```
@misc{lan2024locating,
      title={Locating Cross-Task Sequence Continuation Circuits in Transformers}, 
      author={Michael Lan and Fazl Barez},
      year={2024},
      eprint={2311.04131},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

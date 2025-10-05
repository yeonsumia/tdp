# AntMaze Multi-goal Exploration

## Installation

Set up the environment and install all dependencies as specified in [this link](https://cleandiffuserteam.github.io/CleanDiffuserDocs/docs/introduction/installation/installation.html).

## Pretrain models

Train the AntMaze model:
```bash
cd CleanDiffuser
python pipelines/diffuser_d4rl_antmaze.py task env_name=antmaze-large-diverse-v2 mode=train
```

The trained model will be saved at: `CleanDiffuser/results/diffuser_d4rl_antmaze/antmaze-large-diverse-v2`.

## Run tasks
Run <strong>TDP</strong> on AntMaze Multi-goal Exploration:
```bash
cd CleanDiffuser
python pipelines/diffuser_d4rl_antmaze.py task.env_name=antmaze-large-diverse-ours-v2 mode=inference num_candidates=32 num_envs=1 use_sub_tree=True pg=True
```

# Maze2D Gold Picking

## Installation

```bash
conda env create -f environment.yml
conda activate diffuser
pip install -e .
```

## Download weights

Download the pretrained Maze2D models released by Janner et al. from [this link](https://www.dropbox.com/s/za14rwp8to1bosn/maze2d-logs.zip?e=2&dl=0).

After downloading, place the `maze2d-logs.zip` file in the root directory of the project. Extract the contents with:

```bash
unzip maze2d-logs.zip
```

This will create a new folder (`logs/`) in the root directory.

## Run tasks
Run <strong>TDP</strong> on maze2d-medium:
```bash
python scripts/plan_maze2d.py --dataset maze2d-medium-v1 --task_name maze2d-medium --use_tree --pg
```

Run <strong>TDP</strong> on multi2d-medium:
```bash
python scripts/plan_maze2d.py --dataset maze2d-medium-v1 --task_name multi2d-medium --use_tree --pg
```

Run <strong>TDP</strong> on maze2d-large:
```bash
python scripts/plan_maze2d.py --dataset maze2d-large-v1 --task_name maze2d-large --use_tree --pg
```

Run <strong>TDP</strong> on multi2d-large:
```bash
python scripts/plan_maze2d.py --dataset maze2d-large-v1 --task_name multi2d-large  --use_tree --pg
```
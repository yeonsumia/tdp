# KUKA Robot Arm Manipulation

## Installation

```bash
conda env create -f environment.yml
conda activate diffusion
pip install -e .
```

## Download weights

Download the pretrained Kuka models released by Janner et al. from [this link](https://www.dropbox.com/s/zofqvtkwpmp4v44/metainfo.tar.gz?dl=0).

After downloading, place the `metainfo.tar.gz` file in the root directory of the project. Extract the contents with:

```bash
tar -xzf metainfo.tar.gz
```

This will create two new folders (`kuka_dataset/` and `logs/`) in the root directory.

## Run tasks
Run <strong>TDP</strong> on PnP (*stack*):
```bash
python scripts/conditional_kuka_planning_eval.py --pg --use_sub_tree
```

Run <strong>TDP</strong> on PnP (*place*):
```bash
python scripts/pick_kuka_planning_eval.py --pg --use_sub_tree
```

Run <strong>TDP</strong> on PnWP:
```bash
python scripts/pick_kuka_planning_eval_pnwp.py --pg --use_sub_tree
```

# Flow-matching SVS


## Data Preparation

data directory: `./data/final`

## Train DiffSVS

```python

CUDA_VISIBLE_DEVICES=4 python main.py --base configs/diff_cfm_test.yaml -t --gpus 1

CUDA_VISIBLE_DEVICES=4,5 python main.py \
  --base configs/diff_cfm.yaml  -t \
  --gpus 3

```
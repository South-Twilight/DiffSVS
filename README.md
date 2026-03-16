# DiffSVS
A Flow Matching SVS implementation.

Implemente based on [TCSinger2](https://github.com/AaronZ345/TCSinger2)

## Data Preparation

data directory: `./data/final`

## Train DiffSVS

```python

CUDA_VISIBLE_DEVICES=4 python main.py --base configs/diff_cfm_test.yaml -t --gpus 1

CUDA_VISIBLE_DEVICES=4 python main.py \
  --base configs/diff_cfm.yaml  -t \
  --gpus 1

CUDA_VISIBLE_DEVICES=4 python main.py \
  --base configs/diff_cfm.yaml \
  -t \
  --gpus 1 \
  --resume logs/2026-03-15T12-10-16_diff_cfm/checkpoints/epoch=000004-step=000009725.ckpt

CUDA_VISIBLE_DEVICES=4 python main.py \
  --base configs/diff_cfm.yaml \
  -t \
  --gpus 1 \
  --resume logs/2026-03-15T12-10-16_diff_cfm/checkpoints/epoch=000004-step=000009725.ckpt \
  --name diff_cfm_resume_9725
```
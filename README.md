# DiffSVS
A Flow Matching SVS implementation.

Implemente based on [TCSinger2](https://github.com/AaronZ345/TCSinger2)

## Data Preparation

data directory: `./data/final`

## Train DiffSVS

```python

CUDA_VISIBLE_DEVICES=4 python main.py --base configs/diff_cfm_test.v2.yaml -t --gpus 1

CUDA_VISIBLE_DEVICES=3 python main.py \
  --base configs/diff_cfm.v1.yaml  -t \
  --gpus 1

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
  --base configs/diff_cfm.v1.yaml \
  -t \
  --gpus 4 \
  --resume logs/2026-03-18T07-44-19_resume_2026-03-17T16-24-33_diff_cfm.v1_33550 \
  --name resume_2026-03-17T16-24-33_diff_cfm.v1_33550

CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --base configs/diff_cfm.v2.yaml \
  -t \
  --gpus 2 
```


## Infer DiffSVS

```python

CUDA_VISIBLE_DEVICES=4 python infer.py \
  --config configs/diff_cfm_test.v2.yaml \
  --ckpt logs/2026-03-20T08-34-32_diff_cfm_test.v2/checkpoints/trainstep_checkpoints/epoch=000937-step=000060000.ckpt \
  --manifest_path data/final_test/test.tsv \
  --ddim_steps 25 \
  --scale 1.0 \
  --max_eval 25

```
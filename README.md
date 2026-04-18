# DiffSVS
A Flow Matching SVS implementation.

Implemente based on [TCSinger2](https://github.com/AaronZ345/TCSinger2)

## Data Preparation

data directory: `./data/final`

## Train DiffSVS

```python

CUDA_VISIBLE_DEVICES=2 python main.py --base configs/diff_cfm.test.yaml -t --gpus 1

CUDA_VISIBLE_DEVICES=3 python main.py \
  --base configs/diff_cfm.v1.yaml  -t \
  --gpus 1

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
  --base configs/diff_cfm.v1.yaml \
  -t \
  --gpus 4 \
  --resume logs/2026-03-17T16-24-33_diff_cfm.v1/checkpoints/epoch=000054-step=000033550.ckpt \
  --name resume_2026-03-17T16-24-33_diff_cfm.v1_33550

CUDA_VISIBLE_DEVICES=2,3 python main.py \
  --base configs/diff_cfm.v2.yaml \
  -t \
  --gpus 2 

CUDA_VISIBLE_DEVICES=2,3 python main.py \
  --base configs/diff_cfm.v2.yaml \
  -t \
  --resume exp_ckpt/2026-03-21T13-44-16_diff_cfm.v2 \
  --name resume_2026-03-21T13-44-16_diff_cfm.v2
  --gpus 2 
```


## Infer DiffSVS

```python

CUDA_VISIBLE_DEVICES=6 python infer.py \
  --config configs/diff_cfm.v3.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/final/test_50.tsv \
  --ddim_steps 25 \
  --scale 1.0 \
  --max_eval 0

CUDA_VISIBLE_DEVICES=1 python infer.py \
  --config configs/diff_cfm.v3.yaml \
  --ckpt exp_ckpt/grpo_checkpoints/20260328-093347_cfm_grpo.v5/checkpoints/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v5_ep002_step000126_rank16.pt \
  --manifest_path data/final/test_50.tsv \
  --ddim_steps 25 \
  --scale 1.0 \
  --max_eval 0

CUDA_VISIBLE_DEVICES=5 python infer.py \
  --config configs/diff_cfm.v3.yaml \
  --ckpt exp_ckpt/grpo_checkpoints/20260328-093331_cfm_grpo.v6/checkpoints/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v6_ep002_step000126_rank16.pt \
  --manifest_path data/final/test_50.tsv \
  --ddim_steps 25 \
  --scale 1.0 \
  --max_eval 0

```

## Eval

```sh
# in versa repo

./launch_local.sh \
  /data5/tyx/DiffSVS/exp_outputs/test_50/20260328-093347_cfm_grpo.v5/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v5_ep002_step000126_rank16/wav.scp \
  /data5/tyx/DiffSVS/data/final/gt_test_50.scp \
  /data5/tyx/DiffSVS/exp_outputs/test_50/20260328-093347_cfm_grpo.v5/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v5_ep002_step000126_rank16/eval_results \
  4

```

```python

python utils/rl_utils/avg_eval_results.py \
  --eval_dir /data5/tyx/DiffSVS/exp_outputs/test_50/20260330-094754_cfm_grpo.v1/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v1_ep002_step000126_rank16/eval_results

```

## RL DiffSVS

```python

CUDA_VISIBLE_DEVICES=7 python train_grpo.py \
  --config configs/cfm_grpo_debug.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/rl/train.tsv \
  --max_eval 2

CUDA_VISIBLE_DEVICES=1 python train_grpo.py \
  --config configs/cfm_grpo.v1.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/rl/train.tsv \
  --max_eval 0


CUDA_VISIBLE_DEVICES=6 python train_grpo.py \
  --config configs/cfm_grpo.v6.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/rl/train.tsv \
  --max_eval 0


CUDA_VISIBLE_DEVICES=7 python train_grpo.py \
  --config configs/cfm_grpo.v7.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/rl/train.tsv \
  --max_eval 0


CUDA_VISIBLE_DEVICES=1 python train_grpo.py \
  --config configs/cfm_grpo.v4.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/rl/train.tsv \
  --max_eval 0

CUDA_VISIBLE_DEVICES=2 python train_grpo.py \
  --config configs/cfm_grpo_KL.v4.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/rl/train.tsv \
  --max_eval 0


CUDA_VISIBLE_DEVICES=7 python train_grpo.py \
  --config configs/cfm_grpo.v2.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/rl/train.tsv \
  --max_eval 0

```

## Lora Util

```python

python utils/rl_utils/lora_utils.py \
  --config configs/cfm_grpo.v1.yaml \
  --base_ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --adapter_dir exp_ckpt/grpo_checkpoints/20260330-094754_cfm_grpo.v1/checkpoints/grpo_epoch_1 \
  --output_ckpt exp_ckpt/grpo_checkpoints/20260330-094754_cfm_grpo.v1/checkpoints/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v1_ep001_rank16.pt

```
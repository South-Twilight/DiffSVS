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
  --ckpt exp_ckpt/grpo_checkpoints/20260327-030956_cfm_grpo.v1/checkpoints/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v1_ep002_step000126_rank16.pt \
  --manifest_path data/final_test/test.tsv \
  --ddim_steps 25 \
  --scale 1.0 \
  --max_eval 0

CUDA_VISIBLE_DEVICES=0 python infer.py \
  --config configs/diff_cfm.v3.yaml \
  --ckpt exp_ckpt/grpo_checkpoints/20260327-030454_cfm_grpo.v4/checkpoints/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v4_ep002_step000126_rank16.pt \
  --ddim_steps 25 \
  --scale 1.0 \
  --max_eval 0

CUDA_VISIBLE_DEVICES=4 python infer.py \
  --config configs/diff_cfm.v2.yaml \
  --ckpt /data5/tyx/DiffSVS/exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/final_test/test.tsv \
  --ddim_steps 25 \
  --scale 1.0 \
  --max_eval 0

```

## Eval

```sh
# in versa repo

./launch_local.sh \
  /data5/tyx/DiffSVS/exp_outputs/20260327-030454_cfm_grpo.v4/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v4_ep002_step000126_rank16/wav.scp \
  /data5/tyx/DiffSVS/data/final_test/gt_wav.scp \
  /data5/tyx/DiffSVS/exp_outputs/20260327-030454_cfm_grpo.v4/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v4_ep002_step000126_rank16/eval_results \
  4

```

```python

python utils/rl_utils/avg_eval_results.py \
  --eval_dir /data5/tyx/DiffSVS/exp_outputs/20260327-030454_cfm_grpo.v4/merged_full_policy_base-epoch=000044-step=000057330_cfg-cfm_grpo.v4_ep002_step000126_rank16/eval_results

```

## RL DiffSVS

```python

CUDA_VISIBLE_DEVICES=7 python train_grpo.py \
  --config configs/cfm_grpo_debug.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/rl/train.tsv \
  --max_eval 2

CUDA_VISIBLE_DEVICES=0 python train_grpo.py \
  --config configs/cfm_grpo.v5.yaml \
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


CUDA_VISIBLE_DEVICES=3 python train_grpo.py \
  --config configs/cfm_grpo.v3.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/rl/train.tsv \
  --max_eval 0

CUDA_VISIBLE_DEVICES=2 python train_grpo.py \
  --config configs/cfm_grpo.v4.yaml \
  --ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/epoch=000044-step=000057330.ckpt \
  --manifest_path data/rl/train.tsv \
  --max_eval 0

```

## Lora Util

```python

python utils/lora_utils.py \
  --config configs/cfm_grpo.v1.yaml \
  --base_ckpt exp_ckpt/2026-03-21T15-04-44_diff_cfm.v2/checkpoints/trainstep_checkpoints/epoch=000141-step=000180000.ckpt \
  --adapter_dir exp_ckpt/grpo_checkpoints/20260325-081326_grpo_test_run/checkpoints/grpo_step_120 \
  --output_ckpt exp_ckpt/grpo_checkpoints/20260325-081326_grpo_test_run/merged_step120.pt

```
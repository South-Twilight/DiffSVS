#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import logging
import librosa
import numpy as np
import torch
from espnet2.bin.spk_inference import Speech2Embedding

logger = logging.getLogger(__name__)
def speaker_model_setup(
    model_tag="default", model_path=None, model_config=None, use_gpu=False
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    if model_path is not None and model_config is not None:
        model = Speech2Embedding(
            model_file=model_path, train_config=model_config, device=device
        )
    else:
        if model_tag == "default":
            model_tag = "espnet/voxcelebs12_rawnet3"
        model = Speech2Embedding.from_pretrained(model_tag=model_tag, device=device)
    return model


def speaker_metric(model, pred_x, gt_x, fs):
    pred_arr = np.asarray(pred_x)
    gt_arr = np.asarray(gt_x)
    # Batch mode: [G, T] (allow different T; will align to min length)
    if pred_arr.ndim == 2 and gt_arr.ndim == 2:
        # Align batch size
        if pred_arr.shape[0] != gt_arr.shape[0]:
            if pred_arr.shape[0] == 1:
                pred_arr = np.repeat(pred_arr, gt_arr.shape[0], axis=0)
            elif gt_arr.shape[0] == 1:
                gt_arr = np.repeat(gt_arr, pred_arr.shape[0], axis=0)
            else:
                raise ValueError(
                    f"speaker_metric batch size mismatch: pred_G={pred_arr.shape[0]} gt_G={gt_arr.shape[0]}"
                )
        
        if fs != 16000:
            pred_list = [librosa.resample(x, orig_sr=fs, target_sr=16000) for x in pred_arr]
            gt_list = [librosa.resample(x, orig_sr=fs, target_sr=16000) for x in gt_arr]
            min_len = min(min(len(x) for x in pred_list), min(len(x) for x in gt_list))
            pred_arr = np.stack([x[:min_len] for x in pred_list], axis=0)
            gt_arr = np.stack([x[:min_len] for x in gt_list], axis=0)
        else:
            min_len = min(pred_arr.shape[1], gt_arr.shape[1])
            pred_arr = pred_arr[:, :min_len]
            gt_arr = gt_arr[:, :min_len]

        pred_t = torch.from_numpy(pred_arr).float()
        gt_t = torch.from_numpy(gt_arr).float()
        if pred_t.dim() != 2 or gt_t.dim() != 2:
            raise ValueError(f"speaker_metric batch expects [G, T], got pred={tuple(pred_t.shape)} gt={tuple(gt_t.shape)}")
        # Speech2Embedding.__call__ 仅支持单条（内部会 unsqueeze），batch 需要直接调用底层 spk_model
        pred_t = pred_t.to(model.device)
        gt_t = gt_t.to(model.device)
        pred_emb_t = model.spk_model(speech=pred_t, extract_embd=True)  # [G, D]
        gt_emb_t = model.spk_model(speech=gt_t, extract_embd=True)      # [G, D]
        sim_t = torch.nn.functional.cosine_similarity(pred_emb_t, gt_emb_t, dim=1)
        return {"spk_similarity": sim_t.detach().cpu().numpy().astype(np.float64)}

    # NOTE(jiatong): only work for 16000 Hz
    if fs != 16000:
        gt_arr = librosa.resample(gt_arr, orig_sr=fs, target_sr=16000)
        pred_arr = librosa.resample(pred_arr, orig_sr=fs, target_sr=16000)

    embedding_gen = model(pred_arr).squeeze(0).cpu().numpy()
    embedding_gt = model(gt_arr).squeeze(0).cpu().numpy()
    similarity = np.dot(embedding_gen, embedding_gt) / (
        np.linalg.norm(embedding_gen) * np.linalg.norm(embedding_gt)
    )
    return {"spk_similarity": similarity}


if __name__ == "__main__":
    model = speaker_model_setup()

    # # single
    # a = np.random.random(16000).astype(np.float32)
    # b = np.random.random(16000).astype(np.float32)
    # print("metrics: {}".format(speaker_metric(model, a, b, 16000)))

    # batch
    G = 4
    a_b = np.random.random((G, 16000)).astype(np.float32)
    b_b = np.random.random((G, 16000)).astype(np.float32)
    out_b = speaker_metric(model, a_b, b_b, 16000)
    print("batch metrics:", out_b)

#!/usr/bin/env python3

# Copyright 2023 Takaaki Saeki
# Copyright 2024 Jiatong Shi
# Copyright 2025 Jionghao Han
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import torch
import torchaudio


def pseudo_mos_setup(
    predictor_types, predictor_args, cache_dir="versa_cache", use_gpu=False
):
    # 仅支持 singmos_pro
    predictor_dict = {}
    predictor_fs = {}
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    for predictor in predictor_types:
        if predictor == "singmos_pro":
            torch.hub.set_dir(cache_dir)
            singmos = torch.hub.load(
                "South-Twilight/SingMOS:v1.1.2", "singmos_pro", trust_repo=True
            ).to(device)
            predictor_dict["singmos_pro"] = singmos
            predictor_fs["singmos_pro"] = 16000
        else:
            raise NotImplementedError(
                "Only singmos_pro is supported, got {}".format(predictor)
            )

    return predictor_dict, predictor_fs


def pseudo_mos_metric(pred, fs, predictor_dict, predictor_fs, use_gpu=False):
    pred_arr = np.asarray(pred)
    if pred_arr.ndim == 1:
        batch_pred = pred_arr[None, :]
    elif pred_arr.ndim == 2:
        batch_pred = pred_arr
    else:
        raise ValueError(
            "pred must be shape [T] or [G, T], got {}".format(pred_arr.shape)
        )

    scores = {}
    for predictor in predictor_dict.keys():
        if predictor == "singmos_pro":
            if fs != predictor_fs["singmos_pro"]:
                pred_t = torch.from_numpy(batch_pred).float()
                pred_t = torchaudio.functional.resample(
                    pred_t, orig_freq=fs, new_freq=predictor_fs["singmos_pro"]
                )
                pred_singmos = pred_t.numpy()
            else:
                pred_singmos = batch_pred
            pred_tensor = torch.from_numpy(pred_singmos)
            length_tensor = torch.tensor([pred_tensor.size(1)] * pred_tensor.size(0)).int()
            if use_gpu:
                pred_tensor = pred_tensor.to("cuda")
                length_tensor = length_tensor.to("cuda")
            score = predictor_dict["singmos_pro"](
                pred_tensor.float(), length_tensor
            ).squeeze(-1)
            if pred_tensor.size(0) == 1:
                scores.update(singmos_pro=score[0].item())
            else:
                scores.update(singmos_pro=score.detach().cpu().numpy())
        else:
            raise NotImplementedError(
                "Only singmos_pro is supported, got {}".format(predictor)
            )

    return scores


if __name__ == "__main__":
    a = np.random.random(16000)
    print(a)
    predictor_dict, predictor_fs = pseudo_mos_setup(
        ["singmos_pro"],
        predictor_args={},
    )
    scores = pseudo_mos_metric(
        a, fs=16000, predictor_dict=predictor_dict, predictor_fs=predictor_fs
    )
    print("metrics: {}".format(scores))

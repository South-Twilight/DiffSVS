#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
# Adapted/Inspired by ESPnet/S3PRL-VC from Wen-Chin Huang and Tomoki Hayashi
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pysptk
import pyworld as pw
import scipy
from fastdtw import fastdtw
from scipy.signal import firwin, lfilter


def low_cut_filter(x, fs, cutoff=70):
    """Function to apply low cut filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def spc2npow(spectrogram):
    """Calculate normalized power sequence from spectrogram

    Parameters
    ----------
    spectrogram : array, shape (T, `fftlen / 2 + 1`)
        Array of spectrum envelope

    Return
    ------
    npow : array, shape (`T`, `1`)
        Normalized power sequence

    """

    # frame based processing
    npow = np.apply_along_axis(_spvec2pow, 1, spectrogram)

    meanpow = np.mean(npow)
    npow = 10.0 * np.log10(npow / meanpow)

    return npow


def _spvec2pow(specvec):
    """Convert a spectrum envelope into a power

    Parameters
    ----------
    specvec : vector, shape (`fftlen / 2 + 1`)
        Vector of specturm envelope |H(w)|^2

    Return
    ------
    power : scala,
        Power of a frame

    """

    # set FFT length
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    # specvec is not amplitude spectral |H(w)| but power spectral |H(w)|^2
    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power


def extfrm(data, npow, power_threshold=-20):
    """Extract frame over the power threshold

    Parameters
    ----------
    data: array, shape (`T`, `dim`)
        Array of input data
    npow : array, shape (`T`)
        Vector of normalized power sequence.
    power_threshold : float, optional
        Value of power threshold [dB]
        Default set to -20

    Returns
    -------
    data: array, shape (`T_ext`, `dim`)
        Remaining data after extracting frame
        `T_ext` <= `T`

    """

    T = data.shape[0]
    if T != len(npow):
        raise ("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata


def world_extract(
    x,
    fs,
    f0min,
    f0max,
    mcep_shift=5,
    mcep_fftl=1024,
    mcep_dim=39,
    mcep_alpha=0.466,
    filter_cutoff=70,
):
    # scale from [-1, 1] to [-32768, 32767]
    x = x * np.iinfo(np.int16).max

    if x.ndim > 1:
        x = x[:, 0]
        logging.warning(
            "detect multi-channel data for mcd-f0 caluclation, use first channel"
        )

    x = np.array(x, dtype=np.float64)
    x = low_cut_filter(x, fs, cutoff=filter_cutoff)

    # extract features
    f0, time_axis = pw.harvest(
        x.astype(np.double), fs, f0_floor=f0min, f0_ceil=f0max, frame_period=mcep_shift
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=mcep_fftl)
    ap = pw.d4c(x, f0, time_axis, fs, fft_size=mcep_fftl)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)
    npow = spc2npow(sp)

    return {
        "sp": sp,
        "mcep": mcep,
        "ap": ap,
        "f0": f0,
        "npow": npow,
    }


def mcd_f0(
    pred_x,
    gt_x,
    fs,
    f0min,
    f0max,
    mcep_shift=5,
    mcep_fftl=1024,
    mcep_dim=39,
    mcep_alpha=0.466,
    seq_mismatch_tolerance=0.1,
    power_threshold=-20,
    dtw=False,
    batch_workers=None,
):
    pred_arr = np.asarray(pred_x)
    gt_arr = np.asarray(gt_x)
    # Batch mode: [G, T_pred] vs [G, T_gt]
    # 仅要求样本个数一致，不要求时间长度一致
    if pred_arr.ndim == 2 and gt_arr.ndim == 2 and pred_arr.shape[0] == gt_arr.shape[0]:
        tasks = [
            (
                pred_arr[i],
                gt_arr[i],
                fs,
                f0min,
                f0max,
                mcep_shift,
                mcep_fftl,
                mcep_dim,
                mcep_alpha,
                seq_mismatch_tolerance,
                power_threshold,
                dtw,
            )
            for i in range(pred_arr.shape[0])
        ]
        workers = int(batch_workers) if batch_workers is not None else min(len(tasks), max((os.cpu_count() or 1) // 2, 1))
        with ProcessPoolExecutor(max_workers=max(workers, 1)) as ex:
            results = list(ex.map(_mcd_f0_single_worker, tasks))
        mcd_list = [r["mcd"] for r in results]
        f0rmse_list = [r["f0rmse"] for r in results]
        f0corr_list = [r["f0corr"] for r in results]
        return {
            "mcd": np.asarray(mcd_list, dtype=np.float64),
            "f0rmse": np.asarray(f0rmse_list, dtype=np.float64),
            "f0corr": np.asarray(f0corr_list, dtype=np.float64),
        }

    pred_feats = world_extract(
        pred_arr, fs, f0min, f0max, mcep_shift, mcep_fftl, mcep_dim, mcep_alpha
    )
    gt_feats = world_extract(
        gt_arr, fs, f0min, f0max, mcep_shift, mcep_fftl, mcep_dim, mcep_alpha
    )

    if dtw:
        # VAD & DTW based on power
        pred_mcep_nonsil_pow = extfrm(
            pred_feats["mcep"], pred_feats["npow"], power_threshold=power_threshold
        )
        gt_mcep_nonsil_pow = extfrm(
            gt_feats["mcep"], gt_feats["npow"], power_threshold=power_threshold
        )
        # 如果被 VAD 筛空，则无法做 DTW；回退到 NaN（上层可选择忽略/置 0）
        if pred_mcep_nonsil_pow.shape[0] == 0 or gt_mcep_nonsil_pow.shape[0] == 0:
            logging.warning(
                "Empty nonsil frames for DTW (power_threshold=%s): pred=%d gt=%d",
                str(power_threshold),
                int(pred_mcep_nonsil_pow.shape[0]),
                int(gt_mcep_nonsil_pow.shape[0]),
            )
            return {"mcd": np.nan, "f0rmse": np.nan, "f0corr": np.nan}
        _, path = fastdtw(
            pred_mcep_nonsil_pow,
            gt_mcep_nonsil_pow,
            dist=scipy.spatial.distance.euclidean,
        )
        if len(path) == 0:
            logging.warning("Empty DTW path (power). Return NaN.")
            return {"mcd": np.nan, "f0rmse": np.nan, "f0corr": np.nan}
        twf_pow = np.array(path).T

        # MCD using power-based DTW
        pred_mcep_dtw_pow = pred_mcep_nonsil_pow[twf_pow[0]]
        gt_mcep_dtw_pow = gt_mcep_nonsil_pow[twf_pow[1]]
        diff2sum = np.sum((pred_mcep_dtw_pow - gt_mcep_dtw_pow) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

        # VAD & DTW based on f0
        gt_nonsil_f0_idx = np.where(gt_feats["f0"] > 0)[0]
        pred_nonsil_f0_idx = np.where(pred_feats["f0"] > 0)[0]
        try:
            gt_mcep_nonsil_f0 = gt_feats["mcep"][gt_nonsil_f0_idx]
            pred_mcep_nonsil_f0 = pred_feats["mcep"][pred_nonsil_f0_idx]
            if gt_mcep_nonsil_f0.shape[0] == 0 or pred_mcep_nonsil_f0.shape[0] == 0:
                raise ValueError("Empty nonsil f0 frames")
            _, path = fastdtw(
                pred_mcep_nonsil_f0,
                gt_mcep_nonsil_f0,
                dist=scipy.spatial.distance.euclidean,
            )
            if len(path) == 0:
                raise ValueError("Empty DTW path (f0)")
            twf_f0 = np.array(path).T

            # f0RMSE, f0CORR using f0-based DTW
            pred_f0_dtw = pred_feats["f0"][pred_nonsil_f0_idx][twf_f0[0]]
            gt_f0_dtw = gt_feats["f0"][gt_nonsil_f0_idx][twf_f0[1]]
            f0rmse = np.sqrt(np.mean((pred_f0_dtw - gt_f0_dtw) ** 2))
            f0corr = scipy.stats.pearsonr(pred_f0_dtw, gt_f0_dtw)[0]
        except ValueError as e:
            logging.warning(
                "f0 DTW failed (%s). Set f0rmse/f0corr to NaN. "
                "This might due to unconverged training or empty voiced frames.",
                str(e),
            )
            f0rmse = np.nan
            f0corr = np.nan

    else:
        # Use shorter sequence
        pred_seq_len = len(pred_feats["f0"])
        gt_seq_len = len(gt_feats["f0"])
        min_len = min(pred_seq_len, gt_seq_len)
        assert (pred_seq_len + gt_seq_len - 2 * min_len) / (
            pred_seq_len + gt_seq_len
        ) < seq_mismatch_tolerance, "two input sequence mismatch ratio over threshold {}".format(
            seq_mismatch_tolerance
        )
        diff2sum = np.sum(
            (pred_feats["mcep"][:min_len] - gt_feats["mcep"][:min_len]) ** 2, 1
        )
        mcd = np.mean(10 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        f0rmse = np.sqrt(
            np.mean((pred_feats["f0"][:min_len] - gt_feats["f0"][:min_len]) ** 2)
        )
        f0corr = scipy.stats.pearsonr(
            pred_feats["f0"][:min_len], gt_feats["f0"][:min_len]
        )[0]

    return {
        "mcd": mcd,
        "f0rmse": f0rmse,
        "f0corr": f0corr,
    }


def _mcd_f0_single_worker(args):
    (
        pred_x,
        gt_x,
        fs,
        f0min,
        f0max,
        mcep_shift,
        mcep_fftl,
        mcep_dim,
        mcep_alpha,
        seq_mismatch_tolerance,
        power_threshold,
        dtw,
    ) = args
    return mcd_f0(
        pred_x,
        gt_x,
        fs,
        f0min,
        f0max,
        mcep_shift=mcep_shift,
        mcep_fftl=mcep_fftl,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
        seq_mismatch_tolerance=seq_mismatch_tolerance,
        power_threshold=power_threshold,
        dtw=dtw,
    )


# debug code
if __name__ == "__main__":
    # single
    # a = np.random.random(16000).astype(np.float64)
    # b = np.random.random(16000).astype(np.float64)
    # print("single metrics:", mcd_f0(a, b, 16000, 40, 810, dtw=False))

    # batch
    G = 4
    a_b = np.random.random((G, 16000)).astype(np.float64)
    b_b = np.random.random((G, 16000)).astype(np.float64)
    out_b = mcd_f0(a_b, b_b, 16000, 40, 810, dtw=False, batch_workers=2)
    print("batch metrics:", out_b)

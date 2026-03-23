from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from ldm.dataset.diffsvs_dataset import PHN_PAD_ID, PITCH_PAD_ID, phn_set


def _safe_literal_list(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if not isinstance(v, str):
        return []
    try:
        parsed = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    return []


def _resolve_path(path_value: Any, data_dir: Optional[str]) -> Optional[str]:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    p = path_value.strip()
    if os.path.isabs(p):
        return p
    if data_dir:
        return os.path.join(data_dir, p)
    return p


def _guess_f0_path_from_latent(latent_path: Optional[str]) -> str:
    if not latent_path:
        return ""
    return str(latent_path).replace("latent", "f0").replace(".npy", "_f0.npy")


class DiffSVSInferConditionDataset(Dataset):
    """
    仅用于推理阶段的乐谱条件 Dataset（不加载 latent / f0）：
    返回每条样本的 cond 字段与文件元信息，供 ``infer.py`` 做 CFM 采样。
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        max_eval: int = 0,
        max_duration: float = 20.0,
        data_dir: Optional[str] = None,
        use_singer_map: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.data_dir = data_dir
        self.max_duration = float(max_duration)

        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"找不到 manifest: {manifest_path}")

        df = pd.read_csv(manifest_path, sep="\t")
        if "duration" in df.columns and self.max_duration > 0:
            df = df[df["duration"].astype(float) <= self.max_duration]
        if max_eval and max_eval > 0:
            df = df.iloc[: int(max_eval)]
        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("过滤后数据为空，请检查 manifest/max_eval/max_duration")
        self.dataset = df

        # 推理阶段默认不使用 singer id（固定 0）
        self.use_singer_map = bool(use_singer_map)

        self.ph2id = {p: i for i, p in enumerate(phn_set)}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset.iloc[idx]

        phn_str_list = _safe_literal_list(row.get("ph", "[]"))
        ep_pitches = _safe_literal_list(row.get("ep_pitches", "[]"))
        notedurs = _safe_literal_list(row.get("ep_notedurs", "[]"))
        notetypes = _safe_literal_list(row.get("ep_types", "[]"))

        if not (len(phn_str_list) == len(ep_pitches) == len(notedurs) == len(notetypes)):
            raise ValueError(
                f"manifest 第 {idx} 行 token 字段长度不一致: "
                f"ph={len(phn_str_list)}, pitches={len(ep_pitches)}, "
                f"notedurs={len(notedurs)}, notetypes={len(notetypes)}"
            )
        if len(phn_str_list) == 0:
            raise ValueError(f"manifest 第 {idx} 行为空 token 序列")

        phn_ids = [self.ph2id.get(str(p), PHN_PAD_ID) for p in phn_str_list]
        pitch_ids = [int(x) for x in ep_pitches]
        note_durs = [float(x) for x in notedurs]
        note_types = [int(x) for x in notetypes]

        # 兼容推理脚本常见列名
        item_name = str(row.get("item_name", f"item_{idx}"))
        audio_path = _resolve_path(
            row.get("audio_path", row.get("wav_fn", row.get("wav_path", ""))),
            self.data_dir,
        )
        latent_path = _resolve_path(row.get("latent_path", row.get("mel_path", "")), self.data_dir)
        f0_path = _guess_f0_path_from_latent(latent_path)
        if f0_path and not os.path.isfile(f0_path):
            f0_path = ""

        prompt_latent_path = None
        cand_list = _safe_literal_list(
            row.get("prompt_latent_paths", row.get("prompt_mel_paths", "[]"))
        )
        for p in cand_list:
            rp = _resolve_path(p, self.data_dir)
            if rp and os.path.isfile(rp):
                prompt_latent_path = rp
                break

        return {
            "phn": torch.tensor(phn_ids, dtype=torch.long),
            "pitches": torch.tensor(pitch_ids, dtype=torch.long),
            "notedurs": torch.tensor(note_durs, dtype=torch.float32),
            "notetypes": torch.tensor(note_types, dtype=torch.long),
            "spk_id": torch.tensor(0, dtype=torch.long),
            "infer": True,
            "f_name": item_name,
            "audio_path": audio_path if audio_path is not None else "",
            "latent_path": latent_path if latent_path is not None else "",
            "f0_path": f0_path,
            "prompt_latent_path": prompt_latent_path if prompt_latent_path is not None else "",
        }


def infer_condition_collate_fn(
    batch: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str], List[str], List[str], List[str], List[str]]:
    """
    返回：
    - cond: 推理条件字典（tensor 首维 B）
    - names, audio_paths, latent_paths, f0_paths, prompt_latent_paths: 元信息列表
    """
    if len(batch) == 0:
        raise ValueError("空 batch")

    phn = torch.nn.utils.rnn.pad_sequence(
        [x["phn"] for x in batch], batch_first=True, padding_value=PHN_PAD_ID
    )
    pitches = torch.nn.utils.rnn.pad_sequence(
        [x["pitches"] for x in batch], batch_first=True, padding_value=PITCH_PAD_ID
    )
    notedurs = torch.nn.utils.rnn.pad_sequence(
        [x["notedurs"] for x in batch], batch_first=True, padding_value=0.0
    )
    notetypes = torch.nn.utils.rnn.pad_sequence(
        [x["notetypes"] for x in batch], batch_first=True, padding_value=4
    )
    spk_id = torch.stack([x["spk_id"] for x in batch], dim=0).long()

    cond: Dict[str, Any] = {
        "phn": phn,
        "pitches": pitches,
        "notedurs": notedurs,
        "notetypes": notetypes,
        "spk_id": spk_id,
        "infer": True,
    }

    names = [str(x.get("f_name", "")) for x in batch]
    audio_paths = [str(x.get("audio_path", "")) for x in batch]
    latent_paths = [str(x.get("latent_path", "")) for x in batch]
    f0_paths = [str(x.get("f0_path", "")) for x in batch]
    prompt_latent_paths = [str(x.get("prompt_latent_path", "")) for x in batch]
    return cond, names, audio_paths, latent_paths, f0_paths, prompt_latent_paths


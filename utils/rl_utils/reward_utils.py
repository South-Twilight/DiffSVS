import os
import logging
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def load_reward_yaml_config(path: str):
    """
    读取 configs/reward.yaml（列表结构）并转为按 name 索引的 dict。
    例如:
      - name: mcd_f0
        ...
    转为:
      {"mcd_f0": {...}, ...}
    """
    if path is None or str(path).strip() == "":
        return {}
    if not os.path.exists(path):
        logger.warning("reward config file not found: %s", path)
        return {}
    try:
        cfg = OmegaConf.load(path)
        raw = OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        logger.warning("failed to load reward config file: %s, err=%s", path, str(e))
        return {}
    if not isinstance(raw, list):
        logger.warning("reward config file should be a list, got type=%s", type(raw).__name__)
        return {}
    out = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name", None)
        if not isinstance(name, str) or name.strip() == "":
            continue
        out[name.strip()] = item
    return out
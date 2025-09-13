import os
from typing import Any

import torch


def save_adapter(adapter: Any, out_dir: str, filename: str = "siglip_adapter.pt"):
    """Save the trainable projection layers of the SigLIP adapter.

    Supports both single-head (pool_proj) and two-head (pool_proj_1 + pool_proj_2) pooled
    projections for compatibility with single-encoder mode.
    """
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "hidden_proj": adapter.hidden_proj.state_dict(),
        "projection_dim": int(getattr(getattr(adapter, "config", object()), "projection_dim", 0)),
    }

    # Prefer single pooled head if present; otherwise save two-head variant
    if hasattr(adapter, "pool_proj"):
        payload["pool_proj"] = adapter.pool_proj.state_dict()
        payload["double_pooled"] = False
    elif hasattr(adapter, "pool_proj_1") and hasattr(adapter, "pool_proj_2"):
        payload["pool_proj_1"] = adapter.pool_proj_1.state_dict()
        payload["pool_proj_2"] = adapter.pool_proj_2.state_dict()
        payload["double_pooled"] = True
    else:
        raise AttributeError("Adapter has neither pool_proj nor (pool_proj_1, pool_proj_2)")

    # Best-effort include of underlying SigLIP config for reference
    try:
        payload["siglip_config"] = adapter.model.config.to_dict()
    except Exception:
        payload["siglip_config"] = {}

    torch.save(payload, os.path.join(out_dir, filename))


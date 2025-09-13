#!/usr/bin/env python
import os
import tempfile
import torch
import torch.nn as nn

# Import save_adapter from repo root
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.adapter_io import save_adapter


class DummyConfig:
    def __init__(self, projection_dim: int = 1280):
        self.projection_dim = int(projection_dim)


class DummyModelCfg:
    def __init__(self):
        self._d = {"note": "dummy"}

    def to_dict(self):
        return dict(self._d)


class DummyModel:
    def __init__(self):
        self.config = DummyModelCfg()


class DummyAdapterSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_proj = nn.Linear(8, 16)
        self.pool_proj = nn.Linear(16, 32)
        self.model = DummyModel()
        self.config = DummyConfig(projection_dim=32)


class DummyAdapterDouble(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_proj = nn.Linear(8, 16)
        self.pool_proj_1 = nn.Linear(16, 20)
        self.pool_proj_2 = nn.Linear(16, 12)
        self.model = DummyModel()
        self.config = DummyConfig(projection_dim=32)


def main():
    tmpdir = tempfile.mkdtemp(prefix="smoke_save_")
    single_path = os.path.join(tmpdir, "single.pt")
    double_path = os.path.join(tmpdir, "double.pt")

    a1 = DummyAdapterSingle()
    a2 = DummyAdapterDouble()

    save_adapter(a1, tmpdir, filename="single.pt")
    save_adapter(a2, tmpdir, filename="double.pt")

    s1 = torch.load(single_path, map_location="cpu")
    s2 = torch.load(double_path, map_location="cpu")

    # Minimal assertions / prints
    print("Single keys:", sorted(s1.keys()))
    print("Double keys:", sorted(s2.keys()))
    assert "hidden_proj" in s1 and "pool_proj" in s1 and s1.get("double_pooled") is False
    assert "hidden_proj" in s2 and "pool_proj_1" in s2 and "pool_proj_2" in s2 and s2.get("double_pooled") is True
    print("OK: smoke save produced expected payloads at", tmpdir)


if __name__ == "__main__":
    main()

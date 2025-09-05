# models/datasets.py
"""
SpyNet Dataset Loaders
----------------------
Loads processed artifacts from `data/processed/` produced by Day 2–4:
  - graph_edges.csv       (user/resource interaction edges)
  - sequences.json        (per-user event timestamps)
  - processed_logs.jsonl  (streamed, preprocessed logs)

Provides:
  - GraphDataset (for PyTorch Geometric / GTN)
  - SequenceDataset (for temporal models / N-TPP)
  - BehaviorWindowDataset (for contrastive/tabular models)
  - IDEncoder (stable ID ↔ int mapping with persistence)
  - small helpers: collate functions, safe file IO

All loaders are CPU-friendly; switch to CUDA in training scripts if available.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# PyG is optional at import time (we raise a helpful error on use if missing)
try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.utils import from_scipy_sparse_matrix
except Exception:  # pragma: no cover
    PyGData = None
    from_scipy_sparse_matrix = None

# -----------------------------
# Paths & constants
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
VOCAB_DIR = PROC_DIR / "vocab"
VOCAB_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_EDGES_CSV = PROC_DIR / "graph_edges.csv"
SEQUENCES_JSON = PROC_DIR / "sequences.json"
STREAMED_JSONL = PROC_DIR / "processed_logs.jsonl"


# -----------------------------
# Utilities
# -----------------------------

def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nHint: {hint}"
        )


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass
class IDEncoder:
    """
    Persistent string↔int encoder for stable node/user IDs across runs.
    Saves to data/processed/vocab/<name>.json
    """
    name: str
    _tok2id: Dict[str, int] = None
    _id2tok: List[str] = None

    def __post_init__(self):
        self.path = VOCAB_DIR / f"{self.name}.json"
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self._id2tok = payload["id2tok"]
            self._tok2id = {tok: i for i, tok in enumerate(self._id2tok)}
        else:
            self._tok2id = {}
            self._id2tok = []

    def fit_update(self, tokens: Iterable[str]) -> None:
        for tok in tokens:
            if tok not in self._tok2id:
                self._tok2id[tok] = len(self._id2tok)
                self._id2tok.append(tok)
        self._save()

    def encode(self, tok: str) -> int:
        if tok not in self._tok2id:
            # unseen token -> add on the fly
            self._tok2id[tok] = len(self._id2tok)
            self._id2tok.append(tok)
            self._save()
        return self._tok2id[tok]

    def decode(self, idx: int) -> str:
        return self._id2tok[idx]

    def __len__(self) -> int:
        return len(self._id2tok)

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump({"id2tok": self._id2tok}, f, ensure_ascii=False, indent=2)


# -----------------------------
# GraphDataset (for GTN / GNN)
# -----------------------------

class GraphDataset(Dataset):
    """
    Loads a homogeneous graph from `graph_edges.csv` (comma-separated edge list):
        src_node,dst_node
    Nodes may look like: "user:U123", "device:PC-01", "file:/path/doc.txt"
    We build:
      - node_index mapping via IDEncoder("graph_nodes")
      - edge_index (2, E) tensor
      - simple node features X (degree and one-hot type)
    Returns a single-item dataset: PyG Data object (use dataset[0]).
    """

    def __init__(self, edges_csv: Path = GRAPH_EDGES_CSV):
        if PyGData is None:
            raise ImportError(
                "torch-geometric is required for GraphDataset. "
                "Install PyG wheels compatible with your Torch build."
            )
        _require_file(edges_csv, hint="Run Day-2 preprocessing to generate graph_edges.csv")
        self.edges_csv = edges_csv

        df = pd.read_csv(self.edges_csv, header=None)
        if df.shape[1] < 2:
            raise ValueError("graph_edges.csv must have at least two columns: src,dst")

        self.src_raw = df.iloc[:, 0].astype(str).tolist()
        self.dst_raw = df.iloc[:, 1].astype(str).tolist()
        self.nodes_enc = IDEncoder("graph_nodes")
        self.nodes_enc.fit_update(set(self.src_raw) | set(self.dst_raw))

        # Build edge_index
        src_idx = [self.nodes_enc.encode(s) for s in self.src_raw]
        dst_idx = [self.nodes_enc.encode(d) for d in self.dst_raw]
        self.edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

        # Node features: [degree_norm, one_hot(node_type)]
        node_types = [self._node_type(tok) for tok in self.nodes_enc._id2tok]
        type_vocab = sorted(set(node_types))
        type_to_id = {t: i for i, t in enumerate(type_vocab)}
        degrees = self._degree_counts(len(self.nodes_enc), src_idx, dst_idx)

        deg_feat = torch.tensor(degrees, dtype=torch.float32).unsqueeze(1)
        # min-max normalize degree
        if deg_feat.numel() > 0:
            mn, mx = deg_feat.min(), deg_feat.max()
            if float(mx - mn) > 0:
                deg_feat = (deg_feat - mn) / (mx - mn)

        one_hot = torch.zeros((len(self.nodes_enc), len(type_vocab)), dtype=torch.float32)
        for i, t in enumerate(node_types):
            one_hot[i, type_to_id[t]] = 1.0

        self.x = torch.cat([deg_feat, one_hot], dim=1)

        # Labels are optional (unsupervised). Provide dummy y if needed.
        self.y = None

    def _node_type(self, token: str) -> str:
        # token like "user:U1", "device:PC-1", "file:/path"
        if ":" in token:
            return token.split(":", 1)[0]
        return "unknown"

    def _degree_counts(self, n_nodes: int, src_idx: List[int], dst_idx: List[int]) -> List[float]:
        deg = [0] * n_nodes
        for s, d in zip(src_idx, dst_idx):
            deg[s] += 1
            deg[d] += 1
        return deg

    def __len__(self) -> int:
        # Single graph sample
        return 1

    def __getitem__(self, idx: int):
        assert idx == 0, "GraphDataset contains a single graph. Use dataset[0]."
        data = PyGData(x=self.x, edge_index=self.edge_index)
        return data


# -----------------------------
# SequenceDataset (for N-TPP)
# -----------------------------

class SequenceDataset(Dataset):
    """
    Loads per-user event sequences from `sequences.json`:
      {
        "U123": ["2010-01-01T09:00:00", "2010-01-01T09:05:00", ...],
        ...
      }
    Converts to inter-arrival times (delta t in seconds) with padding.
    Use with a custom collate to batch variable-length sequences.
    """

    def __init__(self, sequences_json: Path = SEQUENCES_JSON, max_len: int = 512):
        _require_file(sequences_json, hint="Run Day-2 preprocessing to generate sequences.json")
        self.max_len = max_len

        with sequences_json.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Filter very short sequences (need at least 2 timestamps to compute deltas)
        self.users: List[str] = []
        self.seqs: List[List[float]] = []

        for user, times in raw.items():
            if len(times) < 2:
                continue
            t = pd.to_datetime(pd.Series(times), utc=False).astype("int64") // 1_000_000_000
            # inter-arrival deltas
            dt = np.diff(t.values).astype(np.float32)
            if len(dt) == 0:
                continue
            # clip/pad
            if len(dt) > max_len:
                dt = dt[-max_len:]
            self.users.append(user)
            self.seqs.append(dt.tolist())

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict:
        dt = self.seqs[idx]
        length = len(dt)
        pad_len = self.max_len - length
        if pad_len > 0:
            dt_padded = dt + [0.0] * pad_len
            mask = [1] * length + [0] * pad_len
        else:
            dt_padded = dt
            mask = [1] * self.max_len
        return {
            "user": self.users[idx],
            "dt": torch.tensor(dt_padded, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "length": torch.tensor(length, dtype=torch.long),
        }


def sequence_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    dt = torch.stack([b["dt"] for b in batch], dim=0)      # (B, T)
    mask = torch.stack([b["mask"] for b in batch], dim=0)  # (B, T)
    length = torch.stack([b["length"] for b in batch], dim=0)  # (B,)
    return {"dt": dt, "mask": mask, "length": length}


# -----------------------------
# BehaviorWindowDataset (tabular windows from streamed logs)
# -----------------------------

class BehaviorWindowDataset(Dataset):
    """
    Builds sliding windows per user from `processed_logs.jsonl`:
      one row per processed event:
        {"user": "...", "event_id": int, "hour": int, "raw_ts": float or None, ...}

    Feature vector per step (simple, extendable):
        [event_id (int), hour (int), delta_minutes (float)]

    Window target is the *next* event_id (can be used for next-event prediction).
    If no next-event available for a window, target = -1.

    Params:
      window_size: number of steps per window
      stride: step between windows
      normalize_delta: z-normalize deltas per user
    """

    def __init__(
        self,
        logs_jsonl: Path = STREAMED_JSONL,
        window_size: int = 16,
        stride: int = 4,
        normalize_delta: bool = True,
    ):
        _require_file(logs_jsonl, hint="Run Day-4 preprocessor to generate processed_logs.jsonl")
        self.window_size = window_size
        self.normalize_delta = normalize_delta

        # Load logs into a per-user list sorted by timestamp
        rows = []
        for obj in load_jsonl(logs_jsonl):
            if "user" not in obj or "event_id" not in obj:
                continue
            ts = obj.get("raw_ts", None)
            if ts is None:
                # fall back to parsed ISO timestamp
                ts_iso = obj.get("timestamp", None)
                ts = float(pd.Timestamp(ts_iso).timestamp()) if ts_iso else None
            rows.append({
                "user": str(obj["user"]),
                "event_id": int(obj["event_id"]),
                "hour": int(obj.get("hour", 0)),
                "ts": float(ts) if ts is not None else math.nan,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("processed_logs.jsonl is empty or missing required fields.")

        df = df.sort_values(["user", "ts"])
        self.windows: List[torch.Tensor] = []
        self.targets: List[int] = []
        self.user_index: List[str] = []

        for user, g in df.groupby("user"):
            g = g.dropna(subset=["ts"])
            if len(g) < self.window_size + 1:
                continue

            # compute deltas in minutes
            ts = g["ts"].values
            dmins = np.diff(ts) / 60.0
            dmins = np.insert(dmins, 0, 0.0)  # first delta = 0 for the first row

            if self.normalize_delta and len(dmins) > 1:
                mu, sd = float(np.mean(dmins[1:])), float(np.std(dmins[1:]) + 1e-6)
                dmins = (dmins - mu) / sd

            feats = np.stack([
                g["event_id"].values.astype(np.float32),
                g["hour"].values.astype(np.float32),
                dmins.astype(np.float32),
            ], axis=1)  # (N, 3)

            # sliding windows
            N = feats.shape[0]
            for start in range(0, N - self.window_size):
                end = start + self.window_size
                x = feats[start:end]             # (W, 3)
                # next event target (classification) if exists
                y = int(g["event_id"].values[end]) if end < N else -1
                self.windows.append(torch.tensor(x, dtype=torch.float32))
                self.targets.append(y)
                self.user_index.append(user)

        if not self.windows:
            raise ValueError("Insufficient events per user to form windows. "
                             "Try reducing window_size or check your data flow.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "x": self.windows[idx],      # (W, 3)
            "y": torch.tensor(self.targets[idx], dtype=torch.long),
            "user": self.user_index[idx],
        }


def behavior_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    X = torch.stack([b["x"] for b in batch], dim=0)   # (B, W, 3)
    y = torch.stack([b["y"] for b in batch], dim=0)   # (B,)
    return {"x": X, "y": y}

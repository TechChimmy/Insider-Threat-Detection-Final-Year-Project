# models/examples/check_datasets.py
"""
Quick checks for SpyNet datasets (run after Day 2–4 are done).

Usage:
  python -m models.examples.check_datasets
"""

from pathlib import Path
from torch.utils.data import DataLoader

from models.datasets import (
    GraphDataset,
    SequenceDataset,
    BehaviorWindowDataset,
    sequence_collate,
    behavior_collate,
    GRAPH_EDGES_CSV,
    SEQUENCES_JSON,
    STREAMED_JSONL,
)

def main():
    print("== SpyNet Dataset Sanity Check ==")

    # 1) Graph (PyG)
    try:
        gds = GraphDataset(GRAPH_EDGES_CSV)
        graph = gds[0]
        print(f"[Graph] nodes={graph.x.shape[0]} edges={graph.edge_index.shape[1]} feat_dim={graph.x.shape[1]}")
    except Exception as e:
        print(f"[Graph] SKIP -> {e}")

    # 2) Sequence (N-TPP)
    try:
        sds = SequenceDataset(SEQUENCES_JSON, max_len=128)
        sdl = DataLoader(sds, batch_size=8, shuffle=True, collate_fn=sequence_collate)
        batch = next(iter(sdl))
        print(f"[Seq] batch dt shape={batch['dt'].shape}, mask shape={batch['mask'].shape}")
    except Exception as e:
        print(f"[Seq] SKIP -> {e}")

    # 3) Behavior windows (tabular)
    try:
        bds = BehaviorWindowDataset(STREAMED_JSONL, window_size=16, stride=4)
        bdl = DataLoader(bds, batch_size=16, shuffle=True, collate_fn=behavior_collate)
        b = next(iter(bdl))
        print(f"[Behavior] batch x shape={b['x'].shape}, y shape={b['y'].shape}")
    except Exception as e:
        print(f"[Behavior] SKIP -> {e}")

    print("Done.")

if __name__ == "__main__":
    main()

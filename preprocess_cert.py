import os
import json
import pandas as pd

RAW_DIR = os.path.join("data", "cert_kaggle")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print(f"[INFO] Looking for raw CSVs in {RAW_DIR}...")
csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
print(f"[INFO] Found {len(csv_files)} CSV files")

# Load CSVs
dfs = []
for f in csv_files:
    path = os.path.join(RAW_DIR, f)
    try:
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"[OK] Loaded {f} with {len(df)} rows")
    except Exception as e:
        print(f"[WARN] Failed to load {f}: {e}")

if not dfs:
    raise FileNotFoundError("No valid CSV files found in data/cert_kaggle/")

df = pd.concat(dfs, ignore_index=True)

# Ensure user column exists
if "user" not in df.columns and "employee" in df.columns:
    df = df.rename(columns={"employee": "user"})

if "event_id" not in df.columns:
    df["event_id"] = range(1, len(df) + 1)

# -------------------------------------------------------------------
# Graph edges
# -------------------------------------------------------------------
print("[INFO] Generating graph_edges.csv...")

edges = []
if "user" in df.columns:
    if "recipient" in df.columns:
        edges = df[["user", "recipient"]].dropna().drop_duplicates()
        edges = edges.rename(columns={"user": "src", "recipient": "dst"})
    elif "to" in df.columns:
        edges = df[["user", "to"]].dropna().drop_duplicates()
        edges = edges.rename(columns={"user": "src", "to": "dst"})

out_file = os.path.join(PROCESSED_DIR, "graph_edges.csv")
if isinstance(edges, pd.DataFrame) and not edges.empty:
    edges.to_csv(out_file, index=False)
    print(f"[OK] Saved graph edges → {out_file}")
else:
    print("[WARN] No suitable edge columns found, skipping graph_edges.csv")

# -------------------------------------------------------------------
# Sequences
# -------------------------------------------------------------------
print("[INFO] Generating sequences.json...")

sequences = {}
if "user" in df.columns and "event_id" in df.columns:
    for uid, group in df.groupby("user"):
        seq = group.sort_values("event_id")["event_id"].tolist()
        sequences[uid] = seq

out_file = os.path.join(PROCESSED_DIR, "sequences.json")
with open(out_file, "w") as f:
    json.dump(sequences, f)
print(f"[OK] Saved sequences → {out_file}")

# -------------------------------------------------------------------
# Behavior features
# -------------------------------------------------------------------
print("[INFO] Generating behavior_features.csv...")

possible_object_cols = ["object", "recipient", "to", "device", "pc"]
object_col = None
for col in possible_object_cols:
    if col in df.columns:
        object_col = col
        break

agg_dict = {"event_id": "count"}
if object_col:
    agg_dict[object_col] = pd.Series.nunique

features = df.groupby("user").agg(agg_dict)

# Rename cleanly
rename_dict = {"event_id": "event_count"}
if object_col:
    rename_dict[object_col] = "unique_objects"
features = features.rename(columns=rename_dict).reset_index()

out_file = os.path.join(PROCESSED_DIR, "behavior_features.csv")
features.to_csv(out_file, index=False)
print(f"[OK] Saved behavior features → {out_file}")

print("[DONE] Preprocessing complete.")

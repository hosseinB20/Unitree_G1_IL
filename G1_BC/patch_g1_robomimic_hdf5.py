import os
import json
import glob
import h5py
import numpy as np
import pandas as pd

dataset_dir = "/home/hosseinbarati/datasets/g1_pickplace/G1_Dex1_PickPlaceCylinder_Dataset_Sim/data/chunk-000"
output_path = "/home/hosseinbarati/g1_pickplace_robomimic.hdf5"
action_scale_path = "/home/hosseinbarati/g1_action_scale.json"

files = sorted(glob.glob(os.path.join(dataset_dir, "*.parquet")))
if not files:
    raise RuntimeError(f"No parquet files found in {dataset_dir}")

# First pass: compute a simple per-dimension max-abs action scale.
all_actions = []
for file in files:
    df = pd.read_parquet(file, columns=["action"])
    actions = np.stack(df["action"].to_list()).astype(np.float32)
    all_actions.append(actions)

all_actions = np.concatenate(all_actions, axis=0)
action_scale = np.max(np.abs(all_actions), axis=0)
action_scale[action_scale < 1e-6] = 1.0

with open(action_scale_path, "w") as f:
    json.dump({"action_scale": action_scale.tolist()}, f, indent=2)

env_args = {
    "env_name": "Isaac-PickPlace-Cylinder-G129-Dex1-Joint",
    "env_type": "custom",
    "env_kwargs": {}
}

with h5py.File(output_path, "w") as h5:
    data = h5.create_group("data")
    data.attrs["env_args"] = json.dumps(env_args)

    total = 0
    for ep_idx, file in enumerate(files):
        print("Processing:", file)
        df = pd.read_parquet(file)

        states = np.stack(df["observation.state"].to_list()).astype(np.float32)
        actions = np.stack(df["action"].to_list()).astype(np.float32)

        # Normalize actions to roughly [-1, 1] for robomimic compatibility.
        actions = np.clip(actions / action_scale, -1.0, 1.0)

        demo = data.create_group(f"demo_{ep_idx}")
        demo.create_dataset("states", data=states)
        demo.create_dataset("actions", data=actions)

        obs = demo.create_group("obs")
        obs.create_dataset("state", data=states)

        next_obs = demo.create_group("next_obs")
        next_obs.create_dataset("state", data=np.concatenate([states[1:], states[-1:]], axis=0))

        demo.attrs["num_samples"] = len(states)
        total += len(states)

    data.attrs["total"] = total

print("Saved:", output_path)
print("Saved action scale:", action_scale_path)

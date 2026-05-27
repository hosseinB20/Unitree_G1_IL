import os
import json

ROOT = "/home/hosseinbarati/datasets/g1_pickplace/G1_Dex1_PickPlaceCylinder_Dataset_Sim"

episodes_file = os.path.join(ROOT, "meta/episodes.jsonl")

manifest = []

with open(episodes_file, "r") as f:
    for line in f:
        item = json.loads(line)

        ep_idx = item["episode_index"]
        task = item["tasks"][0]
        length = item["length"]

        entry = {
            "episode_index": ep_idx,
            "task": task,
            "length": length,

            "parquet_path":
                f"{ROOT}/data/chunk-000/episode_{ep_idx:06d}.parquet",

            "cam_high":
                f"{ROOT}/videos/chunk-000/observation.images.cam_left_high/episode_{ep_idx:06d}.mp4",

            "cam_left_wrist":
                f"{ROOT}/videos/chunk-000/observation.images.cam_left_wrist/episode_{ep_idx:06d}.mp4",

            "cam_right_wrist":
                f"{ROOT}/videos/chunk-000/observation.images.cam_right_wrist/episode_{ep_idx:06d}.mp4"
        }

        manifest.append(entry)

output_path = "episode_manifest.json"

with open(output_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"saved {len(manifest)} episodes to {output_path}")

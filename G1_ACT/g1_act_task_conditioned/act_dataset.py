import json
import random
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class G1ACTDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        stats_path="episodes_stats.jsonl",
        chunk_size=20,
        image_size=128,
        stride=10,
        normalize_state=True,
        normalize_action=True,
    ):
        with open(manifest_path, "r") as f:
            self.episodes = json.load(f)

        self.chunk_size = int(chunk_size)
        self.stride = int(stride)
        self.normalize_state = normalize_state
        self.normalize_action = normalize_action

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.state_mean, self.state_std, self.action_mean, self.action_std = self._load_global_stats(
            stats_path
        )

        # Build sample index over ALL episodes and many windows per episode.
        # stride=1 means every possible window; increase to 4 or 5 if training is too slow.
        self.samples = []
        for ep_idx, ep in enumerate(self.episodes):
            T = int(ep["length"])
            max_start = T - self.chunk_size
            if max_start < 0:
                continue

            for start in range(0, max_start + 1, self.stride):
                self.samples.append((ep_idx, start))

    def _load_global_stats(self, stats_path):
        """
        Compute weighted global mean/std across all episode stats.
        Uses episodes_stats.jsonl if available.
        """
        stats_file = Path(stats_path)
        if not stats_file.exists():
            # Fallback to no normalization
            state_mean = np.zeros(16, dtype=np.float32)
            state_std = np.ones(16, dtype=np.float32)
            action_mean = np.zeros(16, dtype=np.float32)
            action_std = np.ones(16, dtype=np.float32)
            return state_mean, state_std, action_mean, action_std

        with open(stats_file, "r") as f:
            lines = [json.loads(line) for line in f]

        def aggregate(feature_name):
            total_count = 0.0
            total_sum = None
            total_second_moment = None

            for item in lines:
                feat = item["stats"][feature_name]
                mean = np.asarray(feat["mean"], dtype=np.float64)
                std = np.asarray(feat["std"], dtype=np.float64)
                count = float(feat["count"][0] if isinstance(feat["count"], list) else feat["count"])

                var = std ** 2
                second_moment = var + mean ** 2

                if total_sum is None:
                    total_sum = mean * count
                    total_second_moment = second_moment * count
                else:
                    total_sum += mean * count
                    total_second_moment += second_moment * count

                total_count += count

            global_mean = total_sum / total_count
            global_var = np.maximum(total_second_moment / total_count - global_mean ** 2, 1e-8)
            global_std = np.sqrt(global_var)

            return global_mean.astype(np.float32), global_std.astype(np.float32)

        state_mean, state_std = aggregate("observation.state")
        action_mean, action_std = aggregate("action")

        return state_mean, state_std, action_mean, action_std

    def __len__(self):
        return len(self.samples)

    def load_video_frame(self, video_path, frame_idx):
        container = av.open(video_path)

        try:
            for i, frame in enumerate(container.decode(video=0)):
                if i == frame_idx:
                    img = frame.to_ndarray(format="rgb24")
                    img = Image.fromarray(img)
                    return self.image_transform(img)

            raise RuntimeError(
                f"could not read frame {frame_idx} from {video_path}"
            )
        finally:
            container.close()

    def _normalize_state(self, x):
        if not self.normalize_state:
            return x
        return (x - self.state_mean) / (self.state_std + 1e-6)

    def _normalize_action(self, x):
        if not self.normalize_action:
            return x
        return (x - self.action_mean) / (self.action_std + 1e-6)

    def _denormalize_action(self, x):
        if not self.normalize_action:
            return x
        return x * (self.action_std + 1e-6) + self.action_mean

    def __getitem__(self, idx):
        ep_idx, start = self.samples[idx]
        ep = self.episodes[ep_idx]

        df = pd.read_parquet(ep["parquet_path"])

        state = np.stack(df["observation.state"].values).astype(np.float32)
        action = np.stack(df["action"].values).astype(np.float32)

        current_state = state[start]
        action_chunk = action[start : start + self.chunk_size]

        current_state = self._normalize_state(current_state)
        action_chunk = self._normalize_action(action_chunk)

        cam_high = self.load_video_frame(ep["cam_high"], start)
        cam_left = self.load_video_frame(ep["cam_left_wrist"], start)
        cam_right = self.load_video_frame(ep["cam_right_wrist"], start)

        return {
            "state": torch.tensor(current_state, dtype=torch.float32),
            "actions": torch.tensor(action_chunk, dtype=torch.float32),
            "cam_high": cam_high,
            "cam_left": cam_left,
            "cam_right": cam_right,
            "task": ep["task"],
        }


if __name__ == "__main__":
    dataset = G1ACTDataset(
        "episode_manifest.json",
        chunk_size=20,
        stride=10,
    )

    sample = dataset[0]

    print()
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(k, v.shape)
        else:
            print(k, v)

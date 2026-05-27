import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from act_dataset import G1ACTDataset
from act_policy import ACTPolicy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 100
LR = 3e-5
CHUNK_SIZE = 20
STRIDE = 1

CHECKPOINT_PATH = "act_checkpoint.pt"
BEST_CHECKPOINT_PATH = "act_checkpoint_best.pt"

torch.backends.cudnn.benchmark = True


dataset = G1ACTDataset(
    "episode_manifest.json",
    stats_path="episodes_stats.jsonl",
    chunk_size=CHUNK_SIZE,
    image_size=224,
    stride=STRIDE,              # set to 1 to use every possible window
    normalize_state=True,
    normalize_action=True,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4 if DEVICE == "cuda" else 0,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(DEVICE == "cuda"),
)

model = ACTPolicy(
    state_dim=16,
    action_dim=16,
    chunk_size=CHUNK_SIZE,
    hidden_dim=512,
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

criterion = nn.SmoothL1Loss()
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"epoch {epoch}")

    for batch in pbar:

        state = batch["state"].to(DEVICE, non_blocking=True)
        actions_gt = batch["actions"].to(DEVICE, non_blocking=True)
        cam_high = batch["cam_high"].to(DEVICE, non_blocking=True)
        cam_left = batch["cam_left"].to(DEVICE, non_blocking=True)
        cam_right = batch["cam_right"].to(DEVICE, non_blocking=True)
        tasks = batch["task"]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            pred = model(
                state,
                cam_high,
                cam_left,
                cam_right,
                tasks,
            )

            loss = criterion(pred, actions_gt)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_description(f"epoch {epoch} loss {loss.item():.6f}")

    avg_loss = total_loss / max(1, len(loader))
    scheduler.step()

    print()
    print(f"epoch {epoch} avg loss {avg_loss:.6f}")
    print(f"lr: {scheduler.get_last_lr()[0]:.8f}")
    print()

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": {
            "state_dim": 16,
            "action_dim": 16,
            "chunk_size": CHUNK_SIZE,
            "hidden_dim": 512,
            "stride": STRIDE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
        },
        "normalization": {
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "action_mean": dataset.action_mean,
            "action_std": dataset.action_std,
        },
    }

    torch.save(ckpt, CHECKPOINT_PATH)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(ckpt, BEST_CHECKPOINT_PATH)
        print(f"saved new best checkpoint: {BEST_CHECKPOINT_PATH}")

print("training complete")

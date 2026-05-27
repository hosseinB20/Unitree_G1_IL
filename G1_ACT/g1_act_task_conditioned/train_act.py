import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from act_dataset import G1ACTDataset
from act_policy import ACTPolicy


# =========================================================
# DEVICE
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nusing device: {DEVICE}\n")


# =========================================================
# TRAIN CONFIG
# =========================================================

BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-4

CHUNK_SIZE = 20

CHECKPOINT_PATH = "act_checkpoint.pt"
BEST_CHECKPOINT_PATH = "act_checkpoint_best.pt"

LOG_DIR = "logs/tensorboard"


# =========================================================
# TENSORBOARD
# =========================================================

writer = SummaryWriter(LOG_DIR)


# =========================================================
# DATASET
# =========================================================

print("loading dataset...\n")

dataset = G1ACTDataset(
    manifest_path="episode_manifest.json",
    chunk_size=CHUNK_SIZE,
    image_size=128,
)

print(f"dataset size: {len(dataset)}\n")


# =========================================================
# DATALOADER
# =========================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


# =========================================================
# MODEL
# =========================================================

model = ACTPolicy(
    state_dim=16,
    action_dim=16,
    chunk_size=CHUNK_SIZE,
    hidden_dim=512,
).to(DEVICE)

print("\nmodel created\n")


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)


# =========================================================
# LOSS
# =========================================================

criterion = nn.MSELoss()


# =========================================================
# BEST LOSS
# =========================================================

best_loss = float("inf")


# =========================================================
# TRAIN LOOP
# =========================================================

global_step = 0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    pbar = tqdm(loader)

    for batch_idx, batch in enumerate(pbar):

        # -------------------------------------------------
        # batch
        # -------------------------------------------------

        state = batch["state"].to(DEVICE)

        actions_gt = batch["actions"].to(DEVICE)

        cam_high = batch["cam_high"].to(DEVICE)

        cam_left = batch["cam_left"].to(DEVICE)

        cam_right = batch["cam_right"].to(DEVICE)

        tasks = batch["task"]

        # -------------------------------------------------
        # forward
        # -------------------------------------------------

        pred = model(
            state,
            cam_high,
            cam_left,
            cam_right,
            tasks,
        )

        # -------------------------------------------------
        # loss
        # -------------------------------------------------

        loss = criterion(
            pred,
            actions_gt
        )

        # -------------------------------------------------
        # backward
        # -------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # stabilize transformer training
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        # -------------------------------------------------
        # logging
        # -------------------------------------------------

        loss_value = loss.item()

        total_loss += loss_value

        writer.add_scalar(
            "train/batch_loss",
            loss_value,
            global_step,
        )

        pbar.set_description(
            f"epoch {epoch:03d} | loss {loss_value:.6f}"
        )

        global_step += 1

    # =====================================================
    # epoch stats
    # =====================================================

    avg_loss = total_loss / len(loader)

    writer.add_scalar(
        "train/epoch_loss",
        avg_loss,
        epoch,
    )

    print()
    print("=" * 60)
    print(f"epoch {epoch}")
    print(f"avg loss: {avg_loss:.6f}")
    print("=" * 60)
    print()

    # =====================================================
    # save latest checkpoint
    # =====================================================

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "avg_loss": avg_loss,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )

    # =====================================================
    # save best checkpoint
    # =====================================================

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            checkpoint,
            BEST_CHECKPOINT_PATH
        )

        print(
            f"saved new best checkpoint "
            f"(loss={best_loss:.6f})"
        )

# =========================================================
# FINISH
# =========================================================

writer.close()

print("\ntraining complete\n")

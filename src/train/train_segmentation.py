"""Training entry for court segmentation model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.segmentation_dataset import CourtSegmentationDataset
from src.models.unet_segmentation import UNetSegmentation
from src.utils.config import load_yaml_config
from src.utils.seed import set_global_seed


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_from_root(root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_items = 0

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, masks)

        if train_mode:
            loss.backward()
            optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(1, total_items)


def train_segmentation(config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    project_root = Path(data_cfg.get("project_root", ".")).resolve()
    labels_index = _resolve_from_root(project_root, str(data_cfg["labels_index"]))
    image_h = int(data_cfg.get("image_height", 512))
    image_w = int(data_cfg.get("image_width", 512))

    train_ds = CourtSegmentationDataset(
        labels_index_path=labels_index,
        split=data_cfg.get("train_split", "train"),
        project_root=project_root,
        image_size=(image_h, image_w),
        horizontal_flip_prob=float(data_cfg.get("horizontal_flip_prob", 0.5)),
    )
    val_ds = CourtSegmentationDataset(
        labels_index_path=labels_index,
        split=data_cfg.get("val_split", "val"),
        project_root=project_root,
        image_size=(image_h, image_w),
        horizontal_flip_prob=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=True,
    )

    configured_device = str(train_cfg.get("device", _default_device()))
    if configured_device == "cuda" and not torch.cuda.is_available():
        print("[train-seg] cuda requested but unavailable; falling back to cpu")
        configured_device = "cpu"
    device = torch.device(configured_device)
    model = UNetSegmentation(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=int(model_cfg.get("num_classes", 5)),
        base_channels=int(model_cfg.get("base_channels", 32)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg.get("epochs", 20))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/seg")).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save exact config snapshot for reproducibility.
    with (ckpt_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val_loss = float("inf")
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)

        row = {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(
            f"[train-seg] epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
        }
        torch.save(checkpoint, last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_path)

    with (ckpt_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "device": str(device),
        "epochs": epochs,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_path": str((ckpt_dir / "history.json").resolve()),
    }

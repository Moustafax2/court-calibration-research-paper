"""Siamese retrieval training entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.pose_retrieval_dataset import PoseRetrievalPairDataset
from src.losses.contrastive_loss import ContrastiveLoss
from src.models.siamese_pose import SiameseRetrievalModel
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
    model: SiameseRetrievalModel,
    loader: DataLoader,
    criterion: ContrastiveLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total = 0.0
    n = 0
    for x1, x2, y in tqdm(loader, leave=False):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        e1, e2 = model(x1, x2)
        loss = criterion(e1, e2, y)
        if train_mode:
            loss.backward()
            optimizer.step()

        bs = x1.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


def train_retrieval(config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    root = Path(data_cfg.get("project_root", ".")).resolve()
    labels_index = _resolve_from_root(root, str(data_cfg["labels_index"]))
    assignments = _resolve_from_root(root, str(data_cfg["assignments"]))
    templates_dir = _resolve_from_root(root, str(data_cfg["templates_dir"]))
    image_h = int(data_cfg.get("image_height", 256))
    image_w = int(data_cfg.get("image_width", 256))
    num_classes = int(data_cfg.get("num_classes", 4))

    train_ds = PoseRetrievalPairDataset(
        labels_index_path=labels_index,
        assignments_path=assignments,
        templates_dir=templates_dir,
        split=str(data_cfg.get("train_split", "train")),
        project_root=root,
        image_size=(image_h, image_w),
        num_classes=num_classes,
        hard_negative_prob=float(data_cfg.get("hard_negative_prob", 0.6)),
        augment_anchor_prob=float(data_cfg.get("augment_anchor_prob", 0.4)),
    )

    # Validation is optional (can be absent in assignments).
    val_ds = None
    try:
        val_ds = PoseRetrievalPairDataset(
            labels_index_path=labels_index,
            assignments_path=assignments,
            templates_dir=templates_dir,
            split=str(data_cfg.get("val_split", "val")),
            project_root=root,
            image_size=(image_h, image_w),
            num_classes=num_classes,
            hard_negative_prob=0.0,
            augment_anchor_prob=0.0,
        )
    except Exception:
        val_ds = None

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    device_name = str(train_cfg.get("device", _default_device()))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = SiameseRetrievalModel(
        in_channels=int(model_cfg.get("in_channels", num_classes)),
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        base_channels=int(model_cfg.get("base_channels", 32)),
    ).to(device)

    criterion = ContrastiveLoss(margin=float(train_cfg.get("margin", 1.0)))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg.get("epochs", 20))
    ckpt_dir = _resolve_from_root(root, str(train_cfg.get("checkpoint_dir", "checkpoints/retrieval")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with (ckpt_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device) if val_loader else train_loss

        print(
            f"[train-retrieval] epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
        }
        torch.save(state, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, best_path)

    with (ckpt_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "device": str(device),
        "num_train": len(train_ds),
        "num_val": len(val_ds) if val_ds is not None else 0,
        "epochs": epochs,
        "best_val_loss": best_val,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }

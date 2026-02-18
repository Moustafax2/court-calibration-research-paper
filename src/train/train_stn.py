"""STN homography refinement training entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.stn_dataset import STNHomographyDataset
from src.models.stn_homography import STNHomographyRegressor
from src.utils.config import load_yaml_config
from src.utils.seed import set_global_seed


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve(root: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    return (root / pp).resolve()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    n = 0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if target_mean is not None and target_std is not None:
            y = (y - target_mean) / target_std
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        if train:
            loss.backward()
            optimizer.step()
        bs = x.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


def train_stn(config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    root = Path(data_cfg.get("project_root", ".")).resolve()
    labels_index = _resolve(root, str(data_cfg["labels_index"]))
    assignments = _resolve(root, str(data_cfg["assignments"]))
    manifest = _resolve(root, str(data_cfg["manifest"]))
    templates_dir = _resolve(root, str(data_cfg["templates_dir"]))
    template_h = _resolve(root, str(data_cfg["template_homographies"]))

    image_h = int(data_cfg.get("image_height", 256))
    image_w = int(data_cfg.get("image_width", 256))
    train_split = str(data_cfg.get("train_split", "train"))
    val_split = str(data_cfg.get("val_split", "val"))

    train_ds = STNHomographyDataset(
        labels_index_path=labels_index,
        assignments_path=assignments,
        manifest_path=manifest,
        templates_dir=templates_dir,
        template_homographies_path=template_h,
        split=train_split,
        project_root=root,
        image_size=(image_h, image_w),
    )
    val_ds = None
    try:
        val_ds = STNHomographyDataset(
            labels_index_path=labels_index,
            assignments_path=assignments,
            manifest_path=manifest,
            templates_dir=templates_dir,
            template_homographies_path=template_h,
            split=val_split,
            project_root=root,
            image_size=(image_h, image_w),
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

    model = STNHomographyRegressor(
        in_channels=int(model_cfg.get("in_channels", 2)),
        base_channels=int(model_cfg.get("base_channels", 32)),
    ).to(device)
    loss_name = str(train_cfg.get("loss", "smooth_l1")).lower()
    if loss_name == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss(beta=float(train_cfg.get("huber_beta", 1.0)))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    normalize_targets = bool(train_cfg.get("normalize_targets", True))
    target_mean_t: torch.Tensor | None = None
    target_std_t: torch.Tensor | None = None
    target_mean_np = None
    target_std_np = None
    if normalize_targets:
        y_train = train_ds.get_targets_array()
        target_mean_np = y_train.mean(axis=0).astype("float32")
        std_floor = float(train_cfg.get("target_std_floor", 1e-3))
        target_std_np = np.maximum(y_train.std(axis=0).astype("float32"), std_floor)
        target_mean_t = torch.from_numpy(target_mean_np).to(device)
        target_std_t = torch.from_numpy(target_std_np).to(device)

    epochs = int(train_cfg.get("epochs", 20))
    ckpt_dir = _resolve(root, str(train_cfg.get("checkpoint_dir", "checkpoints/stn")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with (ckpt_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            target_mean=target_mean_t,
            target_std=target_std_t,
        )
        val_loss = (
            _run_epoch(
                model,
                val_loader,
                criterion,
                None,
                device,
                target_mean=target_mean_t,
                target_std=target_std_t,
            )
            if val_loader
            else train_loss
        )
        print(
            f"[train-stn] epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
            "target_mean": target_mean_np.tolist() if target_mean_np is not None else None,
            "target_std": target_std_np.tolist() if target_std_np is not None else None,
            "normalize_targets": normalize_targets,
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

#!/usr/bin/env python3
"""Court calibration CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.camera.sports import supported_sports

def cmd_train_seg(args: argparse.Namespace) -> int:
    try:
        from src.train.train_segmentation import train_segmentation
    except ModuleNotFoundError as exc:
        print(f"[train-seg] missing dependency: {exc}")
        print("Install requirements first: pip install -r requirements.txt")
        return 2

    summary = train_segmentation(args.config)
    print("[train-seg] completed")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


def cmd_train_pose(args: argparse.Namespace) -> int:
    print(f"[train-pose] config={args.config}")
    print("Not implemented yet.")
    return 0


def cmd_train_stn(args: argparse.Namespace) -> int:
    print(f"[train-stn] config={args.config}")
    print("Not implemented yet.")
    return 0


def cmd_train_e2e(args: argparse.Namespace) -> int:
    print(f"[train-e2e] config={args.config}")
    print("Not implemented yet.")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    print(f"[eval] config={args.config} ckpt={args.ckpt}")
    print("Not implemented yet.")
    return 0


def cmd_infer_video(args: argparse.Namespace) -> int:
    try:
        from src.infer.run_video import run_video_pipeline
    except ModuleNotFoundError as exc:
        print(f"[infer-video] missing dependency: {exc}")
        print("Install requirements first: pip install -r requirements.txt")
        return 2

    summary = run_video_pipeline(
        input_path=args.input,
        output_path=args.output,
        sport=args.sport,
        ckpt=args.ckpt,
        overlay_alpha=args.overlay_alpha,
        device=args.device,
        max_frames=args.max_frames,
    )
    print("[infer-video] completed")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if args.ckpt is not None:
        print(f"  ckpt: {args.ckpt}")
    return 0


def cmd_validate_data(args: argparse.Namespace) -> int:
    from src.data.validate_dataset import validate_manifest

    summary = validate_manifest(
        manifest_path=args.manifest,
        project_root=args.project_root,
        check_files=not args.skip_file_checks,
    )
    print("[validate-data] summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 1 if summary["num_errors"] > 0 else 0


def cmd_generate_labels(args: argparse.Namespace) -> int:
    try:
        from src.data.generate_labels import generate_labels_from_manifest
    except ModuleNotFoundError as exc:
        print(f"[generate-labels] missing dependency: {exc}")
        print("Install requirements first: pip install -r requirements.txt")
        return 2

    summary = generate_labels_from_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        sport=args.sport,
        project_root=args.project_root,
        overwrite=args.overwrite,
        max_rmse=args.max_rmse,
    )
    print("[generate-labels] summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 1 if summary["num_errors"] > 0 else 0


def cmd_annotate_h(args: argparse.Namespace) -> int:
    try:
        from src.data.annotate_homography import annotate_single_frame
    except ModuleNotFoundError as exc:
        print(f"[annotate-h] missing dependency: {exc}")
        print("Install requirements first: pip install -r requirements.txt")
        return 2

    summary = annotate_single_frame(
        image_path=args.image,
        manifest_path=args.manifest,
        sport=args.sport,
        side=args.side,
        split=args.split,
        video_id=args.video_id,
        frame_index=args.frame_index,
        project_root=args.project_root,
        preview_path=args.preview_out,
    )
    print("[annotate-h] saved")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


def cmd_visualize_label(args: argparse.Namespace) -> int:
    try:
        from src.data.visualize_labels import visualize_label_overlay
    except ModuleNotFoundError as exc:
        print(f"[visualize-label] missing dependency: {exc}")
        print("Install requirements first: pip install -r requirements.txt")
        return 2

    summary = visualize_label_overlay(
        frame_path=args.frame,
        mask_path=args.mask,
        output_path=args.output,
        alpha=args.alpha,
        draw_center_line=not args.no_center_line,
    )
    print("[visualize-label] saved")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


def cmd_import_yolo(args: argparse.Namespace) -> int:
    try:
        from src.data.import_yolo import import_yolo_keypoints_to_manifest
    except ModuleNotFoundError as exc:
        print(f"[import-yolo] missing dependency: {exc}")
        print("Install requirements first: pip install -r requirements.txt")
        return 2

    summary = import_yolo_keypoints_to_manifest(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        manifest_out=args.manifest_out,
        sport=args.sport,
        split=args.split,
        dataset_dir=args.dataset_dir,
        side=args.side,
        side_from_class=args.side_from_class,
        class_id=args.class_id,
        visibility_threshold=args.visibility_threshold,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        project_root=args.project_root,
        append=not args.overwrite_manifest,
    )
    print("[import-yolo] summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 1 if summary["num_errors"] > 0 else 0


def _add_common_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Basketball court calibration pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train_seg = subparsers.add_parser("train-seg", help="Train segmentation model")
    _add_common_config_arg(p_train_seg)
    p_train_seg.set_defaults(func=cmd_train_seg)

    p_train_pose = subparsers.add_parser("train-pose", help="Train siamese pose model")
    _add_common_config_arg(p_train_pose)
    p_train_pose.set_defaults(func=cmd_train_pose)

    p_train_stn = subparsers.add_parser("train-stn", help="Train STN refinement model")
    _add_common_config_arg(p_train_stn)
    p_train_stn.set_defaults(func=cmd_train_stn)

    p_train_e2e = subparsers.add_parser("train-e2e", help="Train full end-to-end model")
    _add_common_config_arg(p_train_e2e)
    p_train_e2e.set_defaults(func=cmd_train_e2e)

    p_eval = subparsers.add_parser("eval", help="Evaluate checkpoint")
    _add_common_config_arg(p_eval)
    p_eval.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    p_eval.set_defaults(func=cmd_eval)

    p_infer = subparsers.add_parser("infer-video", help="Run video inference")
    p_infer.add_argument("--input", type=Path, required=True, help="Input video path")
    p_infer.add_argument("--output", type=Path, required=True, help="Output video path")
    p_infer.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint path.",
    )
    p_infer.add_argument(
        "--sport",
        type=str,
        default="basketball",
        choices=supported_sports(),
        help="Sport mode",
    )
    p_infer.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for processed frames.",
    )
    p_infer.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Overlay alpha used for segmentation visualization.",
    )
    p_infer.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Inference device override. Default auto-select.",
    )
    p_infer.set_defaults(func=cmd_infer_video)

    p_validate = subparsers.add_parser(
        "validate-data", help="Validate frame annotation manifest"
    )
    p_validate.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to JSONL annotation manifest.",
    )
    p_validate.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to resolve relative frame paths.",
    )
    p_validate.add_argument(
        "--skip-file-checks",
        action="store_true",
        help="Skip existence checks for frame paths.",
    )
    p_validate.set_defaults(func=cmd_validate_data)

    p_gen = subparsers.add_parser(
        "generate-labels",
        help="Generate semantic region labels from a JSONL annotation manifest",
    )
    p_gen.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to JSONL annotation manifest.",
    )
    p_gen.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated mask PNGs + labels index.",
    )
    p_gen.add_argument(
        "--sport",
        type=str,
        default="basketball",
        choices=supported_sports(),
        help="Sport mode.",
    )
    p_gen.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to resolve relative frame paths.",
    )
    p_gen.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masks.",
    )
    p_gen.add_argument(
        "--max-rmse",
        type=float,
        default=None,
        help="Optional max reprojection RMSE (pixels) to keep a row.",
    )
    p_gen.set_defaults(func=cmd_generate_labels)

    p_annot = subparsers.add_parser(
        "annotate-h",
        help="Interactively annotate frame correspondences and append homography row",
    )
    p_annot.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Frame image path to annotate.",
    )
    p_annot.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSONL manifest to append to.",
    )
    p_annot.add_argument(
        "--sport",
        type=str,
        default="basketball",
        choices=supported_sports(),
        help="Sport mode.",
    )
    p_annot.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split label for this annotation row.",
    )
    p_annot.add_argument(
        "--side",
        type=str,
        default="left",
        choices=["left", "right", "full"],
        help="Basketball reference-point template side.",
    )
    p_annot.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Optional video identifier.",
    )
    p_annot.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Optional frame index from source video.",
    )
    p_annot.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to resolve relative image/manifest paths.",
    )
    p_annot.add_argument(
        "--preview-out",
        type=Path,
        default=Path("outputs/annotation_preview.jpg"),
        help="Optional path to save projected court preview image.",
    )
    p_annot.set_defaults(func=cmd_annotate_h)

    p_vis = subparsers.add_parser(
        "visualize-label",
        help="Overlay semantic label mask on a frame with legend",
    )
    p_vis.add_argument("--frame", type=Path, required=True, help="Input frame image path.")
    p_vis.add_argument("--mask", type=Path, required=True, help="Input mask image path.")
    p_vis.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/label_overlay.jpg"),
        help="Output visualization image path.",
    )
    p_vis.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha in [0,1].",
    )
    p_vis.add_argument(
        "--no-center-line",
        action="store_true",
        help="Disable center divider line.",
    )
    p_vis.set_defaults(func=cmd_visualize_label)

    p_imp = subparsers.add_parser(
        "import-yolo",
        help="Import YOLO keypoint labels and compute homographies into manifest",
    )
    p_imp.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Single folder containing both images and matching YOLO txt files.",
    )
    p_imp.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory of frame images (optional if --dataset-dir is provided).",
    )
    p_imp.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory of YOLO txt labels (optional if --dataset-dir is provided).",
    )
    p_imp.add_argument("--manifest-out", type=Path, required=True, help="Output JSONL manifest path.")
    p_imp.add_argument(
        "--sport",
        type=str,
        default="basketball",
        choices=supported_sports(),
        help="Sport mode.",
    )
    p_imp.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "train", "val", "test"],
        help="Split mode. 'auto' creates train/val split.",
    )
    p_imp.add_argument(
        "--side",
        type=str,
        default=None,
        choices=["left", "right", "full"],
        help="Force side for all rows. If omitted, use --side-from-class or default left.",
    )
    p_imp.add_argument(
        "--side-from-class",
        type=str,
        default=None,
        help="Class-to-side mapping, e.g. '0:left,1:right'.",
    )
    p_imp.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="If set, only import objects with this class id from each txt line set.",
    )
    p_imp.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.5,
        help="Minimum YOLO keypoint visibility to use a point.",
    )
    p_imp.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio when --split auto.",
    )
    p_imp.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for auto split shuffling.",
    )
    p_imp.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Root used to store relative frame paths in manifest.",
    )
    p_imp.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Overwrite manifest instead of appending.",
    )
    p_imp.set_defaults(func=cmd_import_yolo)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

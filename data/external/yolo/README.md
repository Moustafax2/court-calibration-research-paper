# YOLO Import Drop Zone

Place your pre-annotated data here:
- `data/external/yolo/images/` -> frame images
- `data/external/yolo/labels/` -> YOLO keypoint `.txt` files with matching stems

Example pair:
- `images/frame_000123.jpg`
- `labels/frame_000123.txt`

Expected YOLO row format:
- `class_id cx cy w h kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis ...`

Use importer command from repo root:

```bash
python main.py import-yolo \
  --images-dir data/external/yolo/images \
  --labels-dir data/external/yolo/labels \
  --manifest-out data/annotations.jsonl \
  --sport basketball \
  --split train \
  --side-from-class "0:left,1:right"
```

If all labels use one side, force it with `--side left` or `--side right`.

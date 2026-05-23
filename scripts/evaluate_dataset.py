"""
Evaluate backend/best.pt on the downloaded helmet dataset.

The dataset is expected in PASCAL VOC format:
test-data/helmet-detection-kaggle/
├── images/*.png
└── annotations/*.xml
"""

import argparse
import csv
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from detector import HelmetDetector


DATASET_DIR = ROOT_DIR / "test-data" / "helmet-detection-kaggle"
OUTPUT_DIR = ROOT_DIR / "output" / "test-results"
DETAIL_CSV = OUTPUT_DIR / "dataset_evaluation_details.csv"
SUMMARY_JSON = OUTPUT_DIR / "dataset_evaluation_summary.json"
MODEL_PATH = BACKEND_DIR / "best.pt"
CLASSES = ["With Helmet", "Without Helmet"]


def parse_annotation(xml_path):
    root = ET.parse(xml_path).getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASSES:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        objects.append(
            {
                "class": name,
                "bbox": [
                    int(float(box.findtext("xmin", "0"))),
                    int(float(box.findtext("ymin", "0"))),
                    int(float(box.findtext("xmax", "0"))),
                    int(float(box.findtext("ymax", "0"))),
                ],
            }
        )
    return objects


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_predictions(ground_truth, predictions, iou_threshold):
    matched_gt = set()
    matched_pred = set()
    true_positive = Counter()

    candidates = []
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truth):
            if pred["class"] != gt["class"]:
                continue
            overlap = iou(pred["box_coords"], gt["bbox"])
            if overlap >= iou_threshold:
                candidates.append((overlap, pred_idx, gt_idx, pred["class"]))

    for _, pred_idx, gt_idx, class_name in sorted(candidates, reverse=True):
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        true_positive[class_name] += 1

    gt_counts = Counter(item["class"] for item in ground_truth)
    pred_counts = Counter(item["class"] for item in predictions)
    false_positive = Counter()
    false_negative = Counter()
    for class_name in CLASSES:
        false_positive[class_name] = pred_counts[class_name] - true_positive[class_name]
        false_negative[class_name] = gt_counts[class_name] - true_positive[class_name]

    return true_positive, false_positive, false_negative


def safe_metric(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_DIR), help="PASCAL VOC dataset directory")
    parser.add_argument("--max-images", type=int, default=0, help="0 means evaluate all images")
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    image_paths = sorted(images_dir.glob("*.png"))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = HelmetDetector(model_name=str(MODEL_PATH))

    totals = {
        "tp": Counter(),
        "fp": Counter(),
        "fn": Counter(),
        "gt": Counter(),
        "pred": Counter(),
    }
    rows = []
    started_all = time.perf_counter()

    for image_path in image_paths:
        xml_path = annotations_dir / f"{image_path.stem}.xml"
        ground_truth = parse_annotation(xml_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append(
                {
                    "filename": image_path.name,
                    "status": "decode_error",
                    "gt_total": len(ground_truth),
                    "pred_total": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": len(ground_truth),
                    "latency_ms": 0,
                }
            )
            continue

        started = time.perf_counter()
        result = detector.process_image(image, conf_threshold=args.conf_threshold, iou_threshold=0.45)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        predictions = [
            {"class": item["class"], "box_coords": item["box_coords"]}
            for item in result["detections"]
            if item["class"] in CLASSES
        ]

        tp, fp, fn = match_predictions(ground_truth, predictions, args.iou_threshold)
        gt_counts = Counter(item["class"] for item in ground_truth)
        pred_counts = Counter(item["class"] for item in predictions)

        for class_name in CLASSES:
            totals["tp"][class_name] += tp[class_name]
            totals["fp"][class_name] += fp[class_name]
            totals["fn"][class_name] += fn[class_name]
            totals["gt"][class_name] += gt_counts[class_name]
            totals["pred"][class_name] += pred_counts[class_name]

        rows.append(
            {
                "filename": image_path.name,
                "status": "ok",
                "gt_total": len(ground_truth),
                "pred_total": len(predictions),
                "tp": sum(tp.values()),
                "fp": sum(fp.values()),
                "fn": sum(fn.values()),
                "latency_ms": latency_ms,
            }
        )

    elapsed = time.perf_counter() - started_all
    per_class = {}
    for class_name in CLASSES:
        tp = totals["tp"][class_name]
        fp = totals["fp"][class_name]
        fn = totals["fn"][class_name]
        precision = safe_metric(tp, tp + fp)
        recall = safe_metric(tp, tp + fn)
        per_class[class_name] = {
            "ground_truth": totals["gt"][class_name],
            "predicted": totals["pred"][class_name],
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": safe_metric(2 * precision * recall, precision + recall),
        }

    total_tp = sum(totals["tp"].values())
    total_fp = sum(totals["fp"].values())
    total_fn = sum(totals["fn"].values())
    precision = safe_metric(total_tp, total_tp + total_fp)
    recall = safe_metric(total_tp, total_tp + total_fn)
    summary = {
        "dataset": str(dataset_dir),
        "images_evaluated": len(image_paths),
        "ground_truth_objects": sum(totals["gt"].values()),
        "predicted_objects": sum(totals["pred"].values()),
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_metric(2 * precision * recall, precision + recall),
        "conf_threshold": args.conf_threshold,
        "match_iou_threshold": args.iou_threshold,
        "total_seconds": round(elapsed, 2),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 2),
        "per_class": per_class,
    }

    with DETAIL_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {DETAIL_CSV}")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()

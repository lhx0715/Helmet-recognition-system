"""
Run lightweight performance checks for the helmet detection backend.

The script uses the real best.pt model and synthetic resolution variants of the
project sample image. Results are written to output/test-results for the report.
"""

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from detector import HelmetDetector


OUTPUT_DIR = ROOT_DIR / "output" / "test-results"
CSV_PATH = OUTPUT_DIR / "performance_results.csv"
JSON_PATH = OUTPUT_DIR / "performance_results.json"
CHART_PATH = OUTPUT_DIR / "performance_chart.png"
SAMPLE_IMAGE = ROOT_DIR / "69242e8374224ef4bb3ce64b3303574c.jpeg"
MODEL_PATH = BACKEND_DIR / "best.pt"


def load_sample_image():
    if SAMPLE_IMAGE.exists():
        image = cv2.imread(str(SAMPLE_IMAGE), cv2.IMREAD_COLOR)
        if image is not None:
            return image

    image = np.full((480, 640, 3), 210, dtype=np.uint8)
    cv2.rectangle(image, (220, 120), (420, 420), (80, 80, 80), 3)
    cv2.circle(image, (320, 110), 45, (20, 20, 20), -1)
    return image


def make_variants(image):
    variants = []
    for width, height in [(320, 240), (640, 480), (1280, 720)]:
        resized = cv2.resize(image, (width, height))
        variants.append((f"{width}x{height}", resized))

    variants.append(("640x480_dark", cv2.convertScaleAbs(cv2.resize(image, (640, 480)), alpha=0.45, beta=0)))
    variants.append(("640x480_bright", cv2.convertScaleAbs(cv2.resize(image, (640, 480)), alpha=1.25, beta=45)))
    variants.append(("640x480_blur", cv2.GaussianBlur(cv2.resize(image, (640, 480)), (9, 9), 0)))
    return variants


def draw_chart(rows):
    width, height = 900, 420
    margin_left, margin_bottom, margin_top = 120, 70, 50
    chart_width = width - margin_left - 40
    chart_height = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((30, 18), "Helmet Detection Latency by Input Variant", fill=(30, 30, 30))
    draw.line((margin_left, margin_top, margin_left, margin_top + chart_height), fill=(80, 80, 80), width=2)
    draw.line((margin_left, margin_top + chart_height, width - 30, margin_top + chart_height), fill=(80, 80, 80), width=2)

    max_latency = max((row["avg_latency_ms"] for row in rows), default=1)
    bar_gap = 18
    bar_width = max(28, int((chart_width - bar_gap * (len(rows) + 1)) / max(len(rows), 1)))

    for idx, row in enumerate(rows):
        bar_height = int((row["avg_latency_ms"] / max_latency) * (chart_height - 25))
        x1 = margin_left + bar_gap + idx * (bar_width + bar_gap)
        y1 = margin_top + chart_height - bar_height
        x2 = x1 + bar_width
        y2 = margin_top + chart_height
        draw.rectangle((x1, y1, x2, y2), fill=(36, 132, 108))
        draw.text((x1 - 4, y1 - 18), f"{row['avg_latency_ms']:.1f}", fill=(30, 30, 30))
        draw.text((x1 - 18, y2 + 10), row["variant"], fill=(30, 30, 30))

    image.save(CHART_PATH)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = HelmetDetector(model_name=str(MODEL_PATH))
    image = load_sample_image()
    rows = []

    for variant, sample in make_variants(image):
        detector.process_image(sample)
        latencies = []
        detections = 0
        for _ in range(3):
            started = time.perf_counter()
            result = detector.process_image(sample)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            detections = result["detection_count"]

        avg_seconds = sum(latencies) / len(latencies)
        rows.append(
            {
                "variant": variant,
                "width": sample.shape[1],
                "height": sample.shape[0],
                "avg_latency_ms": round(avg_seconds * 1000, 2),
                "fps": round(1 / avg_seconds if avg_seconds > 0 else 0, 2),
                "detection_count": detections,
            }
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    JSON_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_chart(rows)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {CHART_PATH}")


if __name__ == "__main__":
    main()

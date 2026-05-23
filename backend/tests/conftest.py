import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def make_image_bytes(width=80, height=60, color=(40, 120, 200)):
    image = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


class FakeDetector:
    def process_image(self, image, conf_threshold=0.3, iou_threshold=0.45):
        h, w = image.shape[:2]
        pixel = np.zeros((1, 1, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", pixel)
        assert ok
        return {
            "image_base64": base64.b64encode(buffer.tobytes()).decode("utf-8"),
            "detections": [
                {
                    "class": "helmet",
                    "confidence": 0.91,
                    "bbox": {"x1": 1, "y1": 2, "x2": min(w, 30), "y2": min(h, 40)},
                    "box_coords": [1, 2, min(w, 30), min(h, 40)],
                }
            ],
            "detection_count": 1,
            "classes": {"helmet": 1},
        }


@pytest.fixture()
def client(monkeypatch, tmp_path):
    import main
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "detector", FakeDetector())
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    return TestClient(main.app)

import base64
from pathlib import Path

import numpy as np
import pytest

from detector import HelmetDetector


class FakeXY:
    def __init__(self, values):
        self.values = np.array(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBox:
    cls = [0]
    conf = [0.8765]
    xyxy = [FakeXY([2, 3, 30, 40])]


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, image, conf, iou, verbose):
        self.calls.append({"shape": image.shape, "conf": conf, "iou": iou, "verbose": verbose})
        return [type("Result", (), {"boxes": [FakeBox()], "names": {0: "helmet"}})()]


def make_detector_with_fake_model():
    detector = HelmetDetector.__new__(HelmetDetector)
    detector.model_name = "fake.pt"
    detector.model = FakeModel()
    detector.device = "cpu"
    return detector


def test_missing_local_model_file_raises_clear_error(tmp_path):
    missing_model = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        HelmetDetector(model_name=str(missing_model))


def test_normalize_grayscale_image_to_bgr():
    image = np.zeros((12, 16), dtype=np.uint8)

    normalized = HelmetDetector.normalize_image(image)

    assert normalized.shape == (12, 16, 3)


def test_normalize_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        HelmetDetector.normalize_image(np.array([], dtype=np.uint8))


def test_detect_returns_expected_detection_format():
    detector = make_detector_with_fake_model()
    image = np.zeros((64, 80, 3), dtype=np.uint8)

    annotated, detections = detector.detect(image, conf_threshold=0.4, iou_threshold=0.5)

    assert annotated.shape == image.shape
    assert detections == [
        {
            "class": "helmet",
            "confidence": 0.8765,
            "bbox": {"x1": 2, "y1": 3, "x2": 30, "y2": 40},
            "box_coords": [2, 3, 30, 40],
        }
    ]
    assert detector.model.calls[0]["conf"] == 0.4
    assert detector.model.calls[0]["iou"] == 0.5


def test_process_image_summarizes_classes_and_base64():
    detector = make_detector_with_fake_model()
    image = np.zeros((64, 80, 3), dtype=np.uint8)

    result = detector.process_image(image)

    assert result["detection_count"] == 1
    assert result["classes"] == {"helmet": 1}
    assert result["detections"][0]["class"] == "helmet"
    assert base64.b64decode(result["image_base64"])


def test_image_to_base64_accepts_bgra_image():
    detector = make_detector_with_fake_model()
    image = np.zeros((8, 8, 4), dtype=np.uint8)

    encoded = detector.image_to_base64(image)

    assert isinstance(encoded, str)
    assert base64.b64decode(encoded)


def test_image_to_base64_accepts_uint16_color_image():
    detector = make_detector_with_fake_model()
    image = np.full((8, 8, 3), 4095, dtype=np.uint16)

    normalized = HelmetDetector.normalize_image(image)
    encoded = detector.image_to_base64(image)

    assert normalized.dtype == np.uint8
    assert normalized.shape == (8, 8, 3)
    assert isinstance(encoded, str)
    assert base64.b64decode(encoded)

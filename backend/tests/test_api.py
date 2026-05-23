from conftest import make_image_bytes


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_detect_image_accepts_valid_image(client):
    response = client.post(
        "/detect/image?conf_threshold=0.3&iou_threshold=0.45",
        files={"file": ("sample.jpg", make_image_bytes(), "image/jpeg")},
    )

    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["detection_count"] == 1
    assert data["classes"] == {"helmet": 1}
    assert data["image_base64"]


def test_detect_image_rejects_non_image_file(client):
    response = client.post(
        "/detect/image",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only image files are allowed"


def test_detect_image_rejects_empty_file(client):
    response = client.post(
        "/detect/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_detect_image_rejects_corrupt_image(client):
    response = client.post(
        "/detect/image",
        files={"file": ("broken.jpg", b"broken bytes", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "decode" in response.json()["detail"]


def test_detect_image_rejects_out_of_range_threshold(client):
    response = client.post(
        "/detect/image?conf_threshold=1.2",
        files={"file": ("sample.jpg", make_image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 422


def test_detect_batch_reports_partial_failures(client):
    response = client.post(
        "/detect/batch",
        files=[
            ("files", ("valid.jpg", make_image_bytes(), "image/jpeg")),
            ("files", ("notes.txt", b"not an image", "text/plain")),
        ],
    )

    data = response.json()
    assert response.status_code == 200
    assert data["total_files"] == 2
    assert data["successful"] == 1
    assert data["failed"] == 1
    assert data["results"][0]["status"] == "success"
    assert data["results"][1]["status"] == "error"


def test_detect_image_returns_503_when_detector_missing(client, monkeypatch):
    import main

    monkeypatch.setattr(main, "detector", None)
    response = client.post(
        "/detect/image",
        files={"file": ("sample.jpg", make_image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 503

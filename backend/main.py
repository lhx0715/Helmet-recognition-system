from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from detector import HelmetDetector

app = FastAPI(title="Helmet Detection API")

# 配置CORS，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建上传目录
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_PATH = BASE_DIR / "best.pt"

# 全局初始化检测器（启动时加载模型）
detector = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化检测器"""
    global detector
    try:
        detector = HelmetDetector(model_name=str(MODEL_PATH))
        print("✓ Helmet Detector initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Helmet Detector: {e}")
        raise


def _ensure_detector_ready():
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Detector not initialized. Please restart the server."
        )


async def _read_image_upload(file: UploadFile) -> tuple[bytes, np.ndarray]:
    """Validate and decode an uploaded image."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty")

    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image. Please ensure it's a valid image file."
        )

    return content, image


def _save_upload(filename: str | None, content: bytes) -> Path:
    safe_name = Path(filename or "upload.jpg").name
    file_path = UPLOAD_DIR / f"{uuid4().hex}_{safe_name}"
    file_path.write_bytes(content)
    return file_path


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "message": "Helmet Detection API is running"
    }


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.3, ge=0, le=1, description="置信度阈值 (0-1, 默认0.3)"),
    iou_threshold: float = Query(0.45, ge=0, le=1, description="IOU阈值 (0-1, 默认0.45)")
):
    """
    接收图片文件进行头盔检测
    
    Args:
        file: 上传的图片文件
        conf_threshold: 置信度阈值，默认 0.5
        
    Returns:
        包含检测结果图片 (Base64) 和检测物体列表的 JSON 响应
    """
    try:
        _ensure_detector_ready()
        content, image = await _read_image_upload(file)
        
        # 执行检测
        result = detector.process_image(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        
        # 保存原始上传文件
        file_path = _save_upload(file.filename, content)
        
        return {
            "status": "success",
            "filename": file.filename,
            "filepath": str(file_path),
            "size": len(content),
            "content_type": file.content_type,
            "confidence_threshold": conf_threshold,
            "detection_count": result["detection_count"],
            "classes": result["classes"],
            "detections": result["detections"],
            "image_base64": result["image_base64"],
            "message": f"Detection complete. Found {result['detection_count']} objects."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/detect/batch")
async def detect_batch(
    files: list[UploadFile] = File(...),
    conf_threshold: float = Query(0.3, ge=0, le=1, description="置信度阈值 (0-1, 默认0.3)"),
    iou_threshold: float = Query(0.45, ge=0, le=1, description="IOU阈值 (0-1, 默认0.45)")
):
    """
    批量检测多张图片
    
    Args:
        files: 上传的图片文件列表
        conf_threshold: 置信度阈值，默认 0.5
        
    Returns:
        包含每张图片的检测结果列表
    """
    try:
        _ensure_detector_ready()
        
        results = []
        
        for file in files:
            try:
                content, image = await _read_image_upload(file)
                
                # 执行检测
                result = detector.process_image(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                
                # 保存原始上传文件
                file_path = _save_upload(file.filename, content)
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "filepath": str(file_path),
                    "detection_count": result["detection_count"],
                    "classes": result["classes"],
                    "detections": result["detections"],
                    "image_base64": result["image_base64"]
                })
            
            except HTTPException as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": e.detail
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e)
                })
        
        return {
            "status": "success",
            "total_files": len(files),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

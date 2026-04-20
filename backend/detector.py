"""
头盔检测推理模块
负责加载 YOLOv8 模型和执行推理
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO


class HelmetDetector:
    """YOLOv8 头盔检测推理类"""
    
    def __init__(self, model_name: str = "best.pt"):
        """
        初始化检测器
        
        Args:
            model_name: YOLO 模型文件名或路径，默认使用 best.pt (头盔检测模型)
        """
        self.model_name = model_name
        self.model = None
        self.device = "cpu"  # 可修改为 "cuda" 如有 GPU
        self._load_model()
    
    def _load_model(self):
        """加载 YOLO 模型"""
        try:
            print(f"Loading model: {self.model_name}")
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect(self, image_array: np.ndarray, conf_threshold: float = 0.3, iou_threshold: float = 0.45) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        执行目标检测
        
        Args:
            image_array: OpenCV 格式的图片数组 (BGR)
            conf_threshold: 置信度阈值，默认 0.3 (已优化)
            iou_threshold: IOU阈值，默认 0.45 (已优化)
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: 
                - 绘制检测框的图片数组
                - 检测结果列表，每项包含 {class, confidence, bbox, box_coords}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please initialize HelmetDetector first.")
        
        # 执行推理 - 优化参数以提高检测灵敏度
        results = self.model(image_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # 初始化结果列表
        detections = []
        
        # 绘制检测框
        annotated_image = image_array.copy()
        
        for result in results:
            # 获取检测结果
            boxes = result.boxes
            names = result.names  # 类别名称字典
            
            for idx, box in enumerate(boxes):
                # 提取信息
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
                
                # 类别名称
                class_name = names[cls]
                
                # 添加到检测列表
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": int(xyxy[0]),
                        "y1": int(xyxy[1]),
                        "x2": int(xyxy[2]),
                        "y2": int(xyxy[3])
                    },
                    "box_coords": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                })
                
                # 绘制边界框和标签
                self._draw_box(annotated_image, xyxy, class_name, conf)
        
        return annotated_image, detections
    
    def _draw_box(self, image: np.ndarray, xyxy: np.ndarray, class_name: str, conf: float):
        """
        在图片上绘制检测框
        
        Args:
            image: 图片数组
            xyxy: 边界框坐标 [x1, y1, x2, y2]
            class_name: 类别名称
            conf: 置信度
        """
        x1, y1, x2, y2 = xyxy
        
        # 颜色定义 (BGR 格式)
        if class_name.lower() == "helmet":
            color = (0, 255, 0)  # 绿色 - 头盔
            text_color = (0, 255, 0)
        else:
            color = (0, 0, 255)  # 红色 - 人头/其他
            text_color = (0, 0, 255)
        
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        label = f"{class_name} {conf:.2f}"
        
        # 获取文本大小以绘制背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 绘制标签背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            color,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            image,
            label,
            (x1 + 3, y1 - baseline - 3),
            font,
            font_scale,
            (255, 255, 255),  # 白色文字
            thickness
        )
    
    def image_to_base64(self, image_array: np.ndarray, format: str = "jpg") -> str:
        """
        将 OpenCV 图片数组转换为 Base64 编码字符串
        
        Args:
            image_array: OpenCV 格式的图片数组 (BGR)
            format: 输出格式，'jpg' 或 'png'，默认 'jpg'
            
        Returns:
            str: Base64 编码的图片字符串
        """
        # 将 BGR 转换为 RGB
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # 使用 PIL 转换格式
        pil_image = Image.fromarray(image_rgb)
        
        # 转换为指定格式
        buffer = BytesIO()
        image_format = format.upper() if format.upper() in ["JPG", "PNG"] else "JPEG"
        if image_format == "JPG":
            image_format = "JPEG"
        
        pil_image.save(buffer, format=image_format, quality=95)
        buffer.seek(0)
        
        # 编码为 Base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64
    
    def process_image(self, image_array: np.ndarray, conf_threshold: float = 0.3, iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        完整的图片处理流程
        
        Args:
            image_array: OpenCV 格式的图片数组
            conf_threshold: 置信度阈值，默认 0.3
            iou_threshold: IOU阈值，默认 0.45
            
        Returns:
            Dict 包含:
                - image_base64: 检测结果图片的 Base64 编码
                - detections: 检测物体列表
                - detection_count: 检测到的物体总数
                - classes: 检测到的类别及数量统计
        """
        # 执行检测
        annotated_image, detections = self.detect(image_array, conf_threshold, iou_threshold)
        
        # 图片编码
        image_base64 = self.image_to_base64(annotated_image, format="jpg")
        
        # 统计类别信息
        class_counts = {}
        for detection in detections:
            class_name = detection["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "image_base64": image_base64,
            "detections": detections,
            "detection_count": len(detections),
            "classes": class_counts
        }

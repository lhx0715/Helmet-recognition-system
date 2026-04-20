# 🎓 头盔识别系统 - 全栈项目

一个基础的全栈项目框架，包含 FastAPI 后端和 React Vite 前端。

## 📁 项目结构

```
头盔识别/
├── backend/              # FastAPI 后端
│   ├── main.py          # FastAPI 应用主文件
│   ├── detector.py      # YOLOv8 推理类（HelmetDetector）
│   ├── requirements.txt  # Python 依赖
│   └── uploads/         # 上传的图片存储目录（运行时创建）
│
├── frontend/            # React Vite 前端
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUpload.jsx      # 文件上传组件
│   │   │   └── ImageUpload.css      # 上传组件样式
│   │   ├── App.jsx                  # 主应用组件
│   │   ├── App.css                  # 应用样式
│   │   ├── main.jsx                 # 入口文件
│   │   └── index.css                # 全局样式
│   ├── index.html                   # HTML 模板
│   ├── package.json                 # Node 依赖
│   └── vite.config.js               # Vite 配置
│
└── README.md            # 本文档
```

## 🚀 快速开始

### 前置要求

- Python 3.8+ （后端）
- Node.js 16+ （前端）
- npm 或 yarn

### 后端设置

#### 1. 安装 Python 依赖

```bash
cd backend
pip install -r requirements.txt
```

#### 2. 运行 FastAPI 服务器

```bash
python main.py
```

服务器将在 `http://localhost:8000` 启动。

你可以访问以下地址：
- **API 文档**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc

#### 3. API 端点

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查接口，返回服务状态 |
| POST | `/detect/image` | 上传单张图片进行头盔检测，返回检测结果和 Base64 编码的检测图 |
| POST | `/detect/batch` | 批量上传多张图片进行检测，返回所有图片的检测结果 |

**健康检查示例:**
```bash
curl http://localhost:8000/health
```

**单图片检测示例:**
```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://localhost:8000/detect/image
```

**带置信度阈值的检测:**
```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  "http://localhost:8000/detect/image?conf_threshold=0.6"
```

**响应示例:**
```json
{
  "status": "success",
  "filename": "image.jpg",
  "filepath": "uploads/image.jpg",
  "size": 123456,
  "content_type": "image/jpeg",
  "confidence_threshold": 0.5,
  "detection_count": 5,
  "classes": {
    "person": 2,
    "helmet": 3
  },
  "detections": [
    {
      "class": "person",
      "confidence": 0.9524,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 500
      },
      "box_coords": [100, 200, 300, 500]
    },
    {
      "class": "helmet",
      "confidence": 0.8942,
      "bbox": {
        "x1": 120,
        "y1": 180,
        "x2": 200,
        "y2": 220
      },
      "box_coords": [120, 180, 200, 220]
    }
  ],
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAUA...",
  "message": "Detection complete. Found 5 objects."
}
```

## 🏗️ 架构设计

### 后端架构

项目采用 **模块化设计**，将推理逻辑与 API 接口完全分离：

- **detector.py**: 独立的推理模块，包含 `HelmetDetector` 类
  - 负责加载 YOLOv8 模型
  - 执行图片检测推理
  - 图片绘制和编码等操作
  - 可单独进行单元测试和性能测试

- **main.py**: FastAPI 应用层
  - 处理 HTTP 请求/响应
  - 调用 `HelmetDetector` 进行推理
  - 管理文件上传和 CORS 配置

### HelmetDetector 类 API

```python
from detector import HelmetDetector

# 初始化检测器
detector = HelmetDetector(model_name="yolov8n.pt")

# 执行检测（返回绘制框的图片和检测结果列表）
annotated_image, detections = detector.detect(image_array, conf_threshold=0.5)

# 完整处理流程（推荐使用）
result = detector.process_image(image_array, conf_threshold=0.5)
# 返回: {
#   "image_base64": "...",
#   "detections": [...],
#   "detection_count": 5,
#   "classes": {"person": 2, "helmet": 3}
# }

# 图片编码
base64_str = detector.image_to_base64(image_array, format="jpg")
```

### 前端架构

- **App.jsx**: 主应用，管理上传状态和响应展示
- **ImageUpload.jsx**: 文件上传组件，提供 UI 和文件选择逻辑
- 前端接收 Base64 编码的检测图片，直接显示在页面上

### 数据流

```
上传图片
  ↓
FastAPI /detect/image 接口
  ↓
HelmetDetector.process_image()
  ├─ 图片解码
  ├─ YOLO 推理
  ├─ 绘制检测框
  └─ Base64 编码
  ↓
返回 JSON (包含 Base64 图片 + 检测列表)
  ↓
前端显示检测图片和结果
```

### 前端设置

#### 1. 安装 Node 依赖

```bash
cd frontend
npm install
```

#### 2. 运行开发服务器

```bash
npm run dev
```

前端应用将在 `http://localhost:5173` 启动。

#### 3. 构建生产版本

```bash
npm run build
```

构建输出将在 `dist/` 目录中。

#### 4. 预览生产版本

```bash
npm run preview
```

## 💻 使用说明

### 启动整个应用

#### 方式一：分别运行（推荐开发）

**终端 1 - 后端:**
```bash
cd backend
python main.py
```

**终端 2 - 前端:**
```bash
cd frontend
npm run dev
```

#### 方式二：后台运行

**Windows (PowerShell):**
```powershell
# 启动后端
Start-Process python -ArgumentList "backend/main.py"

# 启动前端
cd frontend
npm run dev
```

**Linux/Mac:**
```bash
# 启动后端（后台）
cd backend && python main.py &

# 启动前端
cd ../frontend && npm run dev
```

### 上传图片进行检测

1. 打开前端应用: http://localhost:5173
2. 点击上传区域或拖拽图片
3. 选择一个图片文件（JPG, PNG, GIF, WebP）
4. 查看后端返回的 JSON 响应

## 📦 依赖说明

### 后端 (Python)

- **fastapi**: Web 框架
- **uvicorn**: ASGI 服务器
- **python-multipart**: 处理文件上传
- **ultralytics**: YOLOv8 目标检测库
- **opencv-python**: 图像处理和绘制
- **torch & torchvision**: 深度学习框架

### 前端 (JavaScript)

- **react**: UI 库
- **react-dom**: React DOM 渲染
- **vite**: 构建工具
- **@vitejs/plugin-react**: React 插件

## 🤖 模型配置

### 可用的 YOLOv8 模型

在 `backend/main.py` 中修改 `detector = HelmetDetector(model_name="...")` 可选择不同大小的模型：

| 模型 | 大小 | 速度 | 精度 | 适用场景 |
|------|------|------|------|---------|
| yolov8n.pt | ~7MB | ⚡⚡⚡ 最快 | ★★★ | 移动端、实时处理 |
| yolov8s.pt | ~23MB | ⚡⚡ 快 | ★★★★ | **推荐：平衡** |
| yolov8m.pt | ~50MB | ⚡ 中等 | ★★★★★ 高 | 服务器、离线分析 |
| yolov8l.pt | ~100MB | 慢 | ★★★★★ 很高 | 高精度需求 |
| yolov8x.pt | ~170MB | 最慢 | ★★★★★ 最高 | 最高精度需求 |

### 自定义模型

如果有自己训练的头盔检测模型，可直接替换：

```python
# 在 backend/main.py 中修改
detector = HelmetDetector(model_name="path/to/your/model.pt")
```

### 推理参数调整

在 API 调用时通过 `conf_threshold` 参数调整：

```bash
# 更严格的检测（只有置信度 > 0.7 才显示）
curl -X POST -F "file=@image.jpg" \
  "http://localhost:8000/detect/image?conf_threshold=0.7"

# 更宽松的检测（置信度 > 0.3 即显示）
curl -X POST -F "file=@image.jpg" \
  "http://localhost:8000/detect/image?conf_threshold=0.3"
```

### GPU 加速

如有 NVIDIA GPU，修改 `backend/detector.py` 以启用 CUDA：

```python
# 第 28 行，改为：
self.device = "cuda"  # 使用 GPU
```

## 🔧 常见问题

### 问题 1: CORS 错误
如果前端无法连接到后端，检查：
- 后端是否在 `http://localhost:8000` 运行
- 后端已配置 CORS 中间件（允许来自 `localhost:5173` 的请求）

### 问题 2: 文件上传失败
检查：
- 后端 `uploads` 目录是否存在（运行时自动创建）
- 文件是否是有效的图片格式
- 硬盘空间是否充足

### 问题 3: 前端页面显示空白
检查：
- Node 版本是否 >= 16
- 是否运行了 `npm install`
- 浏览器控制台是否有错误信息

### 问题 4: Python 包安装失败
尝试升级 pip：
```bash
python -m pip install --upgrade pip
```

### 问题 5: 模型加载失败或首次启动缓慢
- **首次启动会自动下载模型**（~20-200MB，取决于选择的模型）
- 确保网络连接正常
- 模型缓存在 `~/.cache/yolov8/` 目录，重启后会直接加载
- 如果下载卡住，可手动下载模型到项目目录

### 问题 6: 推理速度慢
- 使用更小的模型（yolov8n.pt）
- 启用 GPU 加速（如有 NVIDIA GPU）
- 调整置信度阈值以减少后处理时间
- 检查系统资源（CPU、内存）是否充足

### 问题 7: 前端无法显示检测图片
- 检查后端是否正确返回了 `image_base64` 字段
- 确认图片格式有效（前后端使用 JPG 编码）
- 查看浏览器开发者工具的 Network 标签检查响应

## 📝 环境配置

### 后端配置

在 `backend/main.py` 中可以修改：
- **HOST**: `0.0.0.0`（允许外部访问）或 `localhost`
- **PORT**: 默认 `8000`
- **UPLOAD_DIR**: 上传文件存储目录，默认 `uploads`
- **model_name**: 使用的 YOLOv8 模型，默认 `yolov8n.pt`

在 `backend/detector.py` 中可以修改：
- **self.device**: 推理设备，`"cpu"` 或 `"cuda"`（需要 GPU）
- **检测框颜色**: 自定义绿色（头盔）和红色（其他）的 BGR 值

### 前端配置

在 `frontend/vite.config.js` 中可以修改：
- **PORT**: 默认 `5173`
- **HOST**: 默认 `localhost`

## 🎯 后续功能扩展建议

- [x] 集成深度学习模型进行头盔检测 ✅
- [x] 返回 Base64 编码的检测图片 ✅
- [x] 推理逻辑与 API 分离（独立模块） ✅
- [ ] 添加检测结果实时展示
- [ ] 实现用户认证和权限管理
- [ ] 添加数据库存储检测历史
- [ ] 实现实时处理进度反馈
- [ ] 批量图片处理优化
- [ ] 支持视频流实时检测
- [ ] Web UI 美化和增强
- [ ] Docker 容器化部署
- [ ] 单元测试和集成测试

## 🧪 单元测试（示例）

由于 `HelmetDetector` 是独立模块，可以直接进行单元测试：

```python
# test_detector.py
import cv2
from detector import HelmetDetector

def test_detector_initialization():
    """测试检测器初始化"""
    detector = HelmetDetector(model_name="yolov8n.pt")
    assert detector.model is not None
    print("✓ 检测器初始化成功")

def test_image_detection():
    """测试图片检测"""
    detector = HelmetDetector(model_name="yolov8n.pt")
    
    # 加载测试图片
    image = cv2.imread("path/to/test/image.jpg")
    if image is None:
        raise ValueError("Could not load test image")
    
    # 执行检测
    result = detector.process_image(image, conf_threshold=0.5)
    
    assert "image_base64" in result
    assert "detections" in result
    assert "detection_count" in result
    print(f"✓ 检测成功，找到 {result['detection_count']} 个对象")

def test_base64_encoding():
    """测试 Base64 编码"""
    detector = HelmetDetector(model_name="yolov8n.pt")
    image = cv2.imread("path/to/test/image.jpg")
    
    base64_str = detector.image_to_base64(image, format="jpg")
    assert isinstance(base64_str, str)
    assert len(base64_str) > 0
    print("✓ Base64 编码成功")

if __name__ == "__main__":
    test_detector_initialization()
    test_image_detection()
    test_base64_encoding()
    print("\n✅ 所有测试通过！")
```

运行测试：
```bash
cd backend
python test_detector.py
```

## 📄 许可证

MIT

## 👥 支持

如有问题，请检查项目日志或提交 issue。

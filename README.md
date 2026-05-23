# 电动车骑手头盔检测系统

本项目是“软件测试与质量管理”课程作业原型：基于 YOLOv8 的电动车骑手头盔检测系统。系统支持图片上传、头盔目标检测、标注图展示、检测统计和测试报告生成。

## 项目结构

```text
Helmet-recognition-system/
├── backend/
│   ├── main.py                 # FastAPI 接口
│   ├── detector.py             # YOLOv8 检测封装与图像预处理
│   ├── best.pt                 # 头盔检测模型
│   ├── requirements.txt        # 后端依赖
│   └── tests/                  # pytest 单元与接口测试
├── frontend/
│   ├── src/                    # React 前端
│   └── package.json
├── scripts/
│   ├── start-dev.ps1           # 一键启动前后端
│   ├── stop-dev.ps1            # 停止一键启动的前后端进程
│   ├── run_performance_test.py # 性能测试脚本
│   ├── evaluate_dataset.py     # 外部数据集评估脚本
│   └── generate_test_report.py # Word 测试报告生成脚本
├── start-dev.bat               # Windows 双击启动入口
├── test-data/                  # 外部测试数据集
└── output/
    ├── doc/                    # 生成的 Word 测试报告
    └── test-results/           # 性能 CSV/JSON/图表
```

## 环境要求

- 推荐 Python 3.10 或 3.11。深度学习依赖对过新的 Python 版本可能不稳定。
- Node.js 18+。
- Windows、macOS 或 Linux 均可运行；当前作业环境为 Windows。

## 一键启动前后端

Windows 下推荐直接双击根目录的：

```text
start-dev.bat
```

或者在 PowerShell 中运行：

```powershell
.\scripts\start-dev.ps1
```

脚本会同时启动：

- 后端接口：`http://localhost:8000`
- 前端页面：`http://localhost:5173`
- API 文档：`http://localhost:8000/docs`

脚本默认优先使用 `C:\Program Files\Python311\python.exe` 或 Python 3.10。当前模型依赖 Torch/YOLO，建议不要使用 Python 3.14 启动后端。

如果首次运行需要安装依赖，可以执行：

```powershell
.\scripts\start-dev.ps1 -InstallDeps
```

停止一键启动的服务：

```powershell
.\scripts\stop-dev.ps1
```

也可以直接关闭脚本打开的两个终端窗口。

## 手动启动后端

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

后端默认运行在 `http://localhost:8000`。

可访问：

- `GET http://localhost:8000/health`
- `POST http://localhost:8000/detect/image`
- `POST http://localhost:8000/detect/batch`
- Swagger 文档：`http://localhost:8000/docs`

## 手动启动前端

```powershell
cd frontend
npm install
npm run dev
```

前端默认运行在 `http://localhost:5173`。页面支持选择图片、调整置信度阈值和 IOU 阈值，并展示标注图片、检测总数、类别统计和检测框明细。

## API 示例

```powershell
curl.exe -X POST `
  -F "file=@C:\path\to\image.jpg" `
  "http://localhost:8000/detect/image?conf_threshold=0.3&iou_threshold=0.45"
```

成功响应核心字段：

```json
{
  "status": "success",
  "detection_count": 1,
  "classes": {"helmet": 1},
  "detections": [
    {
      "class": "helmet",
      "confidence": 0.91,
      "bbox": {"x1": 1, "y1": 2, "x2": 30, "y2": 40}
    }
  ],
  "image_base64": "..."
}
```

## 自动化测试

单元测试和接口测试均使用 pytest。核心单元测试使用 Mock 模型，避免真实模型推理速度和识别波动影响自动化结果。

```powershell
cd backend
python -m pytest tests
```

覆盖率命令：

```powershell
cd backend
python -m pytest tests --cov=. --cov-report=term-missing
```

已覆盖场景：

- 检测器模型文件缺失
- 灰度图、BGRA 图、空数组等预处理边界
- Base64 编码和类别统计
- `/health` 健康检查
- 合法图片上传
- 非图片、空文件、损坏图片
- 阈值越界
- 批量上传部分成功部分失败
- 检测器未初始化

## 性能测试

性能脚本使用真实 `backend/best.pt` 模型，生成不同分辨率、暗光、强光和模糊样本，记录平均耗时与 FPS。

```powershell
python scripts/run_performance_test.py
```

输出文件：

- `output/test-results/performance_results.csv`
- `output/test-results/performance_results.json`
- `output/test-results/performance_chart.png`

## 外部数据集测试

已下载 Kaggle `andrewmvd/helmet-detection` 数据集到：

```text
test-data/helmet-detection-kaggle/
├── images/       # 764 张 PNG 图片
└── annotations/  # 764 个 PASCAL VOC XML 标注
```

数据集类别与当前模型一致：

- `With Helmet`
- `Without Helmet`

运行完整数据集评估：

```powershell
C:\Program Files\Python311\python.exe scripts\evaluate_dataset.py
```

输出文件：

- `output/test-results/dataset_evaluation_details.csv`
- `output/test-results/dataset_evaluation_summary.json`

当前完整评估结果：764 张图片、1451 个标注目标，整体 Precision 为 0.718，Recall 为 0.7774，F1 为 0.7465。

## 测试报告

生成 Word 测试报告：

```powershell
python scripts/generate_test_report.py
```

报告输出：

- `output/doc/头盔检测系统测试报告.docx`

如果已经先运行性能测试，报告会自动插入性能数据表和柱状图；如果未运行，报告中会保留提示行，便于后续补跑。

## 鲁棒性说明

后端已对以下异常进行明确处理：

- 模型文件缺失：启动阶段抛出清晰错误。
- 检测器未初始化：接口返回 HTTP 503。
- 非图片文件：接口返回 HTTP 400。
- 空文件：接口返回 HTTP 400。
- 损坏或无法解码的图片：接口返回 HTTP 400。
- 灰度图或四通道图片：检测器统一转换为 BGR 三通道。

## 演示建议

2 到 3 分钟演示视频可按以下顺序录制：

1. 启动后端和前端。
2. 打开前端页面，上传正常图片并展示检测结果。
3. 调整置信度阈值，说明检测结果可能变化。
4. 上传非图片或损坏图片，展示异常处理。
5. 展示 pytest 测试通过结果和 Word 测试报告。

## 后续改进

- 补充更多真实电动车骑手样本，统计准确率、召回率和误检率。
- 支持视频流或摄像头实时检测。
- 增加检测历史记录和可视化统计。
- 引入长期运行稳定性和压力测试。

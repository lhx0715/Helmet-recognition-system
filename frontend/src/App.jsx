import { useState } from 'react'
import ImageUpload from './components/ImageUpload'
import './App.css'

const classLabels = {
  helmet: '佩戴头盔',
  no_helmet: '未佩戴头盔',
  person: '骑手/人员',
  head: '头部'
}

function formatClassName(name) {
  return classLabels[name?.toLowerCase()] || name
}

function App() {
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [confThreshold, setConfThreshold] = useState(0.3)
  const [iouThreshold, setIouThreshold] = useState(0.45)

  const handleUpload = async (file) => {
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const params = new URLSearchParams({
        conf_threshold: confThreshold,
        iou_threshold: iouThreshold
      })

      const res = await fetch(`http://localhost:8000/detect/image?${params}`, {
        method: 'POST',
        body: formData
      })

      if (!res.ok) {
        const errorBody = await res.json().catch(() => null)
        throw new Error(errorBody?.detail || `请求失败，状态码：${res.status}`)
      }

      const data = await res.json()
      setResponse(data)
    } catch (err) {
      setError(err.message || '上传检测过程中发生错误')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="container">
        <header className="app-header">
          <div>
            <p className="eyebrow">软件测试与质量管理课程原型</p>
            <h1>电动车骑手头盔检测系统</h1>
          </div>
          <div className="api-status">FastAPI / YOLOv8</div>
        </header>
        
        <div className="controls">
          <div className="control-group">
            <label>置信度阈值 <span>{confThreshold.toFixed(2)}</span></label>
            <input 
              type="range" 
              min="0.1" 
              max="0.9" 
              step="0.05"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              disabled={loading}
            />
          </div>
          <div className="control-group">
            <label>IOU 阈值 <span>{iouThreshold.toFixed(2)}</span></label>
            <input 
              type="range" 
              min="0.1" 
              max="0.9" 
              step="0.05"
              value={iouThreshold}
              onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
              disabled={loading}
            />
          </div>
        </div>
        
        <ImageUpload onUpload={handleUpload} disabled={loading} />

        {loading && (
          <div className="status loading">
            正在调用模型检测，请稍候...
          </div>
        )}

        {error && (
          <div className="status error">
            检测失败：{error}
          </div>
        )}

        {response && (
          <div className="response">
            <div className="result-header">
              <h2>检测结果</h2>
              <div className="stats">
                <div className="stat">
                  <span className="label">目标总数</span>
                  <span className="value">{response.detection_count}</span>
                </div>
                {response.classes && Object.entries(response.classes).map(([className, count]) => (
                  <div key={className} className={`stat ${className.toLowerCase()}`}>
                    <span className="label">{formatClassName(className)}</span>
                    <span className="value">{count}</span>
                  </div>
                ))}
              </div>
            </div>

            {response.image_base64 && (
              <div className="result-image">
                <h3>标注图片</h3>
                <img 
                  src={`data:image/jpeg;base64,${response.image_base64}`} 
                  alt="头盔检测标注结果"
                />
              </div>
            )}

            <div className="detections-list">
              <h3>检测明细</h3>
              <div className="detections-container">
                {response.detections?.length > 0 ? response.detections.map((detection, idx) => (
                  <div 
                    key={idx} 
                    className={`detection-item ${detection.class.toLowerCase()}`}
                  >
                    <div className="detection-class">{formatClassName(detection.class)}</div>
                    <div className="detection-confidence">
                      置信度：{(detection.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="detection-bbox">
                      坐标：({detection.bbox.x1}, {detection.bbox.y1}) - ({detection.bbox.x2}, {detection.bbox.y2})
                    </div>
                  </div>
                )) : (
                  <div className="empty-result">未检测到目标，建议更换清晰图片或降低置信度阈值。</div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

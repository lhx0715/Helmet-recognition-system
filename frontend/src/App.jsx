import { useState } from 'react'
import ImageUpload from './components/ImageUpload'
import './App.css'

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
        throw new Error(`HTTP error! status: ${res.status}`)
      }

      const data = await res.json()
      setResponse(data)
    } catch (err) {
      setError(err.message || 'An error occurred during upload')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="container">
        <h1>🎓 Helmet Detection System</h1>
        
        <div className="controls">
          <div className="control-group">
            <label>Confidence Threshold: <span>{confThreshold.toFixed(2)}</span></label>
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
            <label>IOU Threshold: <span>{iouThreshold.toFixed(2)}</span></label>
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
            ⏳ Processing...
          </div>
        )}

        {error && (
          <div className="status error">
            ❌ Error: {error}
          </div>
        )}

        {response && (
          <div className="response">
            <div className="result-header">
              <h2>Detection Results</h2>
              <div className="stats">
                <div className="stat">
                  <span className="label">Total Objects:</span>
                  <span className="value">{response.detection_count}</span>
                </div>
                {response.classes && Object.entries(response.classes).map(([className, count]) => (
                  <div key={className} className={`stat ${className.toLowerCase()}`}>
                    <span className="label">{className}:</span>
                    <span className="value">{count}</span>
                  </div>
                ))}
              </div>
            </div>

            {response.image_base64 && (
              <div className="result-image">
                <h3>Annotated Image (Green=Helmet, Red=No Helmet):</h3>
                <img 
                  src={`data:image/jpeg;base64,${response.image_base64}`} 
                  alt="Detection Result" 
                />
              </div>
            )}

            <div className="detections-list">
              <h3>Detection Details:</h3>
              <div className="detections-container">
                {response.detections && response.detections.map((detection, idx) => (
                  <div 
                    key={idx} 
                    className={`detection-item ${detection.class.toLowerCase()}`}
                  >
                    <div className="detection-class">{detection.class}</div>
                    <div className="detection-confidence">
                      Confidence: {(detection.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="detection-bbox">
                      Bbox: ({detection.bbox.x1}, {detection.bbox.y1}) - ({detection.bbox.x2}, {detection.bbox.y2})
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

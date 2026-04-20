import { useRef, useState } from 'react'
import './ImageUpload.css'

function ImageUpload({ onUpload, disabled }) {
  const fileInputRef = useRef(null)
  const [preview, setPreview] = useState(null)

  const handleFileSelect = (event) => {
    const file = event.target.files?.[0]
    if (file) {
      // 显示预览
      const reader = new FileReader()
      reader.onload = (e) => {
        setPreview(e.target.result)
      }
      reader.readAsDataURL(file)

      // 上传文件
      onUpload(file)
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="upload-container">
      <div className="upload-area" onClick={handleClick}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          disabled={disabled}
          style={{ display: 'none' }}
        />
        <div className="upload-content">
          <div className="upload-icon">📸</div>
          <h3>Click to Upload Image</h3>
          <p>Supported formats: JPG, PNG, GIF, WebP</p>
        </div>
      </div>

      {preview && (
        <div className="preview-container">
          <h4>Image Preview:</h4>
          <img src={preview} alt="Preview" className="preview-image" />
        </div>
      )}
    </div>
  )
}

export default ImageUpload

import { useRef, useState } from 'react'
import './ImageUpload.css'

function ImageUpload({ onUpload, disabled }) {
  const fileInputRef = useRef(null)
  const [preview, setPreview] = useState(null)
  const [selectedName, setSelectedName] = useState('')

  const handleFileSelect = (event) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedName(file.name)
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
      <button className="upload-area" type="button" onClick={handleClick} disabled={disabled}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          disabled={disabled}
          style={{ display: 'none' }}
        />
        <div className="upload-content">
          <div className="upload-icon" aria-hidden="true">IMG</div>
          <h3>{disabled ? '正在检测图片' : '选择或更换检测图片'}</h3>
          <p>支持 JPG、PNG、GIF、WebP；测试异常场景时可上传非图片或空文件。</p>
          {selectedName && <span className="file-name">{selectedName}</span>}
        </div>
      </button>

      {preview && (
        <div className="preview-container">
          <h4>原图预览</h4>
          <img src={preview} alt="上传图片预览" className="preview-image" />
        </div>
      )}
    </div>
  )
}

export default ImageUpload

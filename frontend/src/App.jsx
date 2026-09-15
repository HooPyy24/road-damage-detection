import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [resultData, setResultData] = useState(null)
  const [videoResultUrl, setVideoResultUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0) // State เก็บ % การอัปโหลด
  const [isDragOver, setIsDragOver] = useState(false)

  // State สำหรับจัดการ Alert Modal Popup (กรณีไม่ใช่ภาพ/วิดีโอถนน)
  const [modalConfig, setModalConfig] = useState({
    isOpen: false,
    title: '',
    message: '',
    imageSrc: null
  })

  const showModal = (title, message, imageSrc = null) => {
    setModalConfig({ isOpen: true, title, message, imageSrc })
  }

  const closeModal = () => {
    setModalConfig({ isOpen: false, title: '', message: '', imageSrc: null })
  }

  const handleFile = (file) => {
    if (file && (file.type.startsWith('image/') || file.type.startsWith('video/'))) {
      setSelectedFile(file)
      setPreview(URL.createObjectURL(file))
      setResultData(null)
      setVideoResultUrl(null)
    } else {
      showModal('ไฟล์ไม่ถูกต้อง', 'กรุณาเลือกไฟล์รูปภาพหรือวิดีโอเท่านั้นครับ')
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = () => setIsDragOver(false)

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      return showModal('แจ้งเตือน', 'กรุณาเลือกรูปภาพหรือวิดีโอก่อนครับ')
    }

    setLoading(true)
    setUploadProgress(0) // รีเซ็ต Progress เป็น 0%
    setResultData(null)
    setVideoResultUrl(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    const isVideo = selectedFile.type.startsWith('video/')

    // Config สำหรับติดตาม Progress ในการส่งไฟล์
    const axiosConfig = {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          setUploadProgress(percentCompleted)
        }
      }
    }

    try {
      if (isVideo) {
        const response = await axios.post('http://127.0.0.1:8000/predict-video', formData, {
          ...axiosConfig,
          responseType: 'blob'
        })

        if (response.data.type === 'application/json') {
          const text = await response.data.text()
          const json = JSON.parse(text)
          if (json.is_valid === false) {
            showModal(
              'ไม่ใช่ไฟล์วิดีโอถนน',
              json.message || 'วิดีโอที่อัปโหลดไม่ใช่เนื้อหาเกี่ยวกับถนน กรุณาอัปโหลดใหม่อีกครั้ง'
            )
            return
          }
        }

        const videoBlobUrl = URL.createObjectURL(response.data)
        setVideoResultUrl(videoBlobUrl)

      } else {
        const response = await axios.post('http://127.0.0.1:8000/predict', formData, axiosConfig)

        if (response.data.is_valid === false) {
          showModal(
            'ภาพที่อัปโหลดไม่ใช่ภาพถนน',
            response.data.message || 'ระบบตรวจพบว่าภาพนี้ไม่ใช่ภาพถนน กรุณาอัปโหลดภาพถนนใหม่อีกครั้ง',
            preview
          )
          return
        }

        setResultData(response.data)
      }
    } catch (error) {
      console.error('Error uploading file:', error)
      showModal('เกิดข้อผิดพลาด', 'ไม่สามารถเชื่อมต่อกับ Server ได้ กรุณาลองใหม่อีกครั้ง')
    } finally {
      setLoading(false)
      setUploadProgress(0)
    }
  }

  const handleDownload = () => {
    const isVideo = selectedFile?.type.startsWith('video/')
    const link = document.createElement('a')
    
    if (isVideo && videoResultUrl) {
      link.href = videoResultUrl
      link.download = `detection_video_${Date.now()}.mp4`
    } else if (resultData?.image_base64) {
      link.href = resultData.image_base64
      link.download = `detection_result_${Date.now()}.jpg`
    } else {
      return
    }
    
    link.click()
  }

  const isVideoFile = selectedFile?.type.startsWith('video/')

  return (
    <div className="app-wrapper">
      <div className="top-header">
        <h1>
          <svg className="header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 19L11 5" />
            <path d="M20 19L13 5" />
            <path d="M12 8v2" />
            <path d="M12 14v2" />
          </svg>
          ระบบวิเคราะห์ความเสียหายบนพื้นผิวถนน
        </h1>
        <p>Road Damage Detection Platform · Powered by YOLOv8s AI</p>
      </div>

      <div className="main-container">
        <div className="info-box">
          <span className="info-title">💡 คำแนะนำ:</span>
          <span>ถ่ายหรือเลือกภาพ/วิดีโอถนน "เน้นกลางภาพ" จัดให้รอยแตกหรือหลุมอยู่ในเฟรมหลัก เพื่อความแม่นยำสูงสุด</span>
        </div>

        <div className="upload-card">
          <h2>อัปโหลดรูปภาพหรือวิดีโอถนน</h2>

          <div 
            className={`dropzone ${loading ? 'disabled' : ''}`}
            style={{ borderColor: isDragOver ? '#38bdf8' : '#0284c7' }}
            onDragOver={!loading ? handleDragOver : undefined}
            onDragLeave={!loading ? handleDragLeave : undefined}
            onDrop={!loading ? handleDrop : undefined}
            onClick={() => !loading && document.getElementById('fileInput').click()}
          >
            <input 
              id="fileInput"
              type="file" 
              accept="image/*,video/*" 
              style={{ display: 'none' }}
              disabled={loading}
              onChange={(e) => handleFile(e.target.files[0])} 
            />
            
            <div className="camera-icon">📷📹</div>
            
            {selectedFile ? (
              <div className="file-name-tag">
                📄 {selectedFile.name}
              </div>
            ) : (
              <>
                <div className="dropzone-text">คลิกหรือลากรูปภาพ/วิดีโอมาวางที่นี่</div>
                <div className="dropzone-subtext">รองรับ PNG, JPG, MP4, AVI, MOV</div>
              </>
            )}
          </div>

          <button 
            className="btn-submit"
            onClick={handleUpload} 
            disabled={loading || !selectedFile}
          >
            🔍 {loading ? (isVideoFile ? 'กำลังส่งข้อมูลวิดีโอ...' : 'กำลังวิเคราะห์ด้วย AI...') : 'ตรวจจับความเสียหาย'}
          </button>
        </div>

        {((resultData && resultData.is_valid) || videoResultUrl) && (
          <div className="results-section">
            <div className="section-title">📊 สรุปผลการตรวจจับ</div>
            
            {resultData && resultData.is_valid && (
              <div className="summary-grid">
                <div className="summary-card total">
                  <div className="num">{resultData.total_detected}</div>
                  <div className="lbl">ความเสียหายทั้งหมด</div>
                </div>
                {Object.entries(resultData.summary).map(([clsName, count]) => (
                  <div key={clsName} className="summary-card">
                    <div className="num">{count}</div>
                    <div className="lbl">{clsName}</div>
                  </div>
                ))}
              </div>
            )}

            <div className="image-grid">
              {preview && (
                <div className="image-card">
                  <div className="image-card-header">
                    <h3>📷 ต้นฉบับ ({isVideoFile ? 'วิดีโอ' : 'รูปภาพ'})</h3>
                  </div>
                  {isVideoFile ? (
                    <video src={preview} controls style={{ width: '100%', borderRadius: '8px' }} />
                  ) : (
                    <img src={preview} alt="Original" />
                  )}
                </div>
              )}

              <div className="image-card">
                <div className="image-card-header">
                  <h3>🎯 ผลการตรวจจับ (YOLOv8)</h3>
                </div>
                {isVideoFile ? (
                  videoResultUrl && <video src={videoResultUrl} controls style={{ width: '100%', borderRadius: '8px' }} />
                ) : (
                  resultData && <img src={resultData.image_base64} alt="Detection Result" />
                )}
                <button className="btn-download" onClick={handleDownload}>
                  ⬇️ ดาวน์โหลด{isVideoFile ? 'วิดีโอ' : 'รูปภาพ'}ผลลัพธ์
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="footer">
          Road Damage Detection System · Computer Vision AI Project
        </div>
      </div>

      {/* --- Progress Modal (เด้งล็อกหน้าจอตอนกำลังส่งไฟล์) --- */}
      {loading && (
        <div className="progress-overlay">
          <div className="progress-card">
            <div className="spinner-icon">⏳</div>
            <h3>กำลังอัปโหลดและวิเคราะห์ผล...</h3>
            <p className="progress-subtext">
              {isVideoFile 
                ? 'วิดีโอใช้เวลาประมวลผลนานกว่าปกติ กรุณาอย่าปิดหรือรีเฟรชหน้าเว็บ' 
                : 'กำลังส่งข้อมูลและให้ AI (YOLOv8) ตรวจจับความเสียหาย'}
            </p>

            <div className="progress-bar-bg">
              <div 
                className="progress-bar-fill" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>

            <div className="progress-percentage-text">
              {uploadProgress < 100 
                ? `กำลังอัปโหลดไฟล์: ${uploadProgress}%` 
                : 'อัปโหลดเสร็จแล้ว! กำลังประมวลผลด้วย AI...'}
            </div>
          </div>
        </div>
      )}

      {/* --- Custom Alert Modal (แจ้งเตือนรูปไม่ใช่ถนน) --- */}
      {modalConfig.isOpen && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-icon">🚫</div>
            <h3>{modalConfig.title}</h3>
            <p>{modalConfig.message}</p>
            
            {modalConfig.imageSrc && (
              <div className="modal-image-container">
                <img src={modalConfig.imageSrc} alt="Non-road uploaded" className="modal-preview-img" />
              </div>
            )}

            <button className="btn-modal-close" onClick={closeModal}>
              ตกลง / ลองใหม่อีกครั้ง
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
import React, { useRef, useEffect, useState } from 'react';

const RealtimeDetector = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const sessionRef = useRef(null);
  const isDetectingRef = useRef(false);

  // ชื่อ Class ตามที่เทรนใน YOLOv8 (ปรับชื่อให้ตรงกับของระบบคุณได้เลยครับ)
  const labels = ['D00', 'D10', 'D20', 'D40']; 

  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsLoading(true);
        const ort = window.ort;
        if (!ort) throw new Error("ไม่พบไลบรารี ONNX Runtime");

        ort.env.wasm.numThreads = 1;
        const session = await ort.InferenceSession.create('/best.onnx', {
          executionProviders: ['wasm'],
        });

        sessionRef.current = session;
        setIsLoading(false);
        console.log("ONNX Model Loaded Successfully!");
      } catch (e) {
        console.error("Failed to load ONNX model:", e);
        setErrorMsg(`โหลดโมเดลไม่สำเร็จ: ${e.message}`);
        setIsLoading(false);
      }
    };

    loadModel();

    return () => {
      isDetectingRef.current = false;
    };
  }, []);

  const startCamera = async () => {
    if (!videoRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 640 }
      });
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play();
        setIsCameraOpen(true);
        isDetectingRef.current = true;
        detectFrame();
      };
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("ไม่พบกล้อง หรือยังไม่ได้กด Start Virtual Camera ใน OBS");
    }
  };

  // ฟังก์ชันแปลงภาพจาก Canvas เป็น Tensor ให้ YOLOv8
  const preprocess = (ctx) => {
    const imgData = ctx.getImageData(0, 0, 640, 640);
    const { data } = imgData;
    const red = [], green = [], blue = [];

    for (let i = 0; i < data.length; i += 4) {
      red.push(data[i] / 255.0);
      green.push(data[i + 1] / 255.0);
      blue.push(data[i + 2] / 255.0);
    }

    const float32Data = new Float32Array([...red, ...green, ...blue]);
    return new window.ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
  };

  // ลูปประมวลผล Real-time
  const detectFrame = async () => {
    if (!isDetectingRef.current || !videoRef.current || !canvasRef.current || !sessionRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (video.readyState === video.HAVE_ENOUGH_DATA) {
      // 1. วาดภาพจากกล้องลง Canvas
      ctx.drawImage(video, 0, 0, 640, 640);

      try {
        // 2. แปลงภาพเป็น Tensor และรันโมเดล YOLO
        const inputTensor = preprocess(ctx);
        const outputMap = await sessionRef.current.run({ images: inputTensor });
        const output = Object.values(outputMap)[0].data;

        // 3. วาดกรอบข้อความ Bounding Box บน Canvas
        // (ส่วนนี้จะดึงพิกัดเพื่อวาดเส้นสีแดง/เหลืองบนภาพถนนครับ)
      } catch (err) {
        console.error("Inference Error:", err);
      }
    }

    requestAnimationFrame(detectFrame);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', padding: '20px' }}>
      {isLoading && <p style={{ color: '#38bdf8', fontSize: '18px' }}>⏳ กำลังโหลดโมเดล Real-time (ONNX)...</p>}
      
      {errorMsg && <p style={{ color: '#ef4444', fontWeight: 'bold' }}>❌ {errorMsg}</p>}

      {!isLoading && !errorMsg && !isCameraOpen && (
        <button 
          onClick={startCamera}
          style={{
            backgroundColor: '#0284c7',
            color: '#fff',
            fontWeight: 'bold',
            padding: '12px 28px',
            borderRadius: '8px',
            border: 'none',
            cursor: 'pointer',
            fontSize: '16px',
            boxShadow: '0 4px 6px -1px rgba(0,0,0,0.2)'
          }}
        >
          📹 เปิดกล้องตรวจจับ Real-time
        </button>
      )}

      <div style={{ 
        position: 'relative', 
        width: '640px', 
        height: '640px', 
        backgroundColor: '#0f172a', 
        borderRadius: '12px', 
        overflow: 'hidden',
        border: '1px solid #334155'
      }}>
        <video ref={videoRef} style={{ display: 'none' }} muted playsInline />
        <canvas ref={canvasRef} width={640} height={640} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
      </div>
    </div>
  );
};

export default RealtimeDetector;
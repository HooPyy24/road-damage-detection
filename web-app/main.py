import cv2
import numpy as np
import tempfile
import os
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# 1. ตั้งค่า CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. โหลด MobileNetV3 Classifier
classifier = models.mobilenet_v3_small()
in_features = classifier.classifier[3].in_features
classifier.classifier[3] = nn.Linear(in_features, 2)

# โหลด Weights ที่เทรนได้ (มั่นใจว่าไฟล์ road_classifier.pth อยู่ในโฟลเดอร์เดียวกับ main.py)
classifier.load_state_dict(torch.load('road_classifier.pth', map_location=device))
classifier.to(device)
classifier.eval()

# Transform สำหรับแปลงภาพเข้า MobileNetV3
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. โหลด YOLOv8 Detector
yolo_model = YOLO("best.pt") # หรือ Path โมเดล YOLOv8 ของคุณ

def check_is_road(cv2_img) -> bool:
    """ฟังก์ชันคัดกรองรูปภาพด้วย MobileNetV3"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    tensor_img = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier(tensor_img)
        # แปลงเป็น ค่าความน่าจะเป็น (Softmax)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Class 0: non_road, Class 1: road
        road_prob = probabilities[1].item()
        non_road_prob = probabilities[0].item()

    # พิมพ์ดูค่าเพื่อ Debug ใน Terminal
    print(f"[MobileNetV3] Non-Road Prob: {non_road_prob:.4f} | Road Prob: {road_prob:.4f}")

    # ต้องเป็น Class 1 (road) และมีความน่าจะเป็นมากกว่า 70% (0.7) ถึงจะยอมให้ผ่าน
    return (road_prob > non_road_prob) and (road_prob >= 0.7)

# --- Endpoint สำหรับรูปภาพ ---
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # คัดกรองภาพก่อนเข้า YOLO
    if not check_is_road(img):
        return {
            "is_valid": False,
            "message": "ภาพที่อัปโหลดไม่ใช่ภาพถนน กรุณาอัปโหลดภาพถนนใหม่อีกครั้ง",
            "total_detected": 0,
            "summary": {},
            "image_base64": None
        }

    # ถ้ารูปผ่านการคัดกรอง ให้ประมวลผลด้วย YOLOv8
    results = yolo_model(img)
    annotated_frame = results[0].plot()

    summary = {}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        summary[cls_name] = summary.get(cls_name, 0) + 1

    _, buffer = cv2.imencode('.jpg', annotated_frame)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return {
        "is_valid": True,
        "total_detected": len(results[0].boxes),
        "summary": summary,
        "image_base64": f"data:image/jpeg;base64,{img_str}"
    }

# --- Endpoint สำหรับวิดีโอ ---
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_in.write(await file.read())
    temp_in.close()

    output_path = temp_in.name.replace(".mp4", "_out.mp4")

    cap = cv2.VideoCapture(temp_in.name)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # อ่านเฟรมแรกมาเช็คก่อนว่าเป็นวิดีโอเกี่ยวกับถนนหรือไม่
    ret, first_frame = cap.read()
    if not ret or not check_is_road(first_frame):
        cap.release()
        out.release()
        if os.path.exists(temp_in.name): os.remove(temp_in.name)
        if os.path.exists(output_path): os.remove(output_path)
        return {
            "is_valid": False,
            "message": "วิดีโอที่อัปโหลดไม่ใช่เนื้อหาเกี่ยวกับถนน กรุณาตรวจสอบอีกครั้ง"
        }

    # เขียนเฟรมแรกเข้าไฟล์ผลลัพธ์
    results = yolo_model(first_frame)
    out.write(results[0].plot())

    # ประมวลผลเฟรมที่เหลือ
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    
    if os.path.exists(temp_in.name):
        os.remove(temp_in.name)

    return FileResponse(
        output_path, 
        media_type="video/mp4", 
        filename="processed_video.mp4"
    )
@app.get("/")
async def root():
    return {"status": "online", "message": "Road Damage Detection API is running"}

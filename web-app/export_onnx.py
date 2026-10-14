from ultralytics import YOLO

# 1. โหลดโมเดลจากไฟล์ best.pt ที่อยู่ในโฟลเดอร์เดียวกัน
model = YOLO('best.pt') 

# 2. แปลงเป็นไฟล์ .onnx
model.export(format='onnx', simplify=True)

print("แปลงไฟล์เป็น best.onnx สำเร็จแล้ว!")
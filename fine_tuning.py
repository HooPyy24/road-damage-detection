import os
import torch

# ป้องกันปัญหา CUDA Driver crash บน GTX 1050 Ti
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from ultralytics import YOLO

if __name__ == '__main__':
    # เคลียร์ memory ค้างใน GPU ก่อนเริ่ม
    torch.cuda.empty_cache()

    # 1. ดึงความเก่งเดิมจาก exp4 มาเป็นจุดเริ่มต้น (Pretrained Base)
    model = YOLO("C:/Project-test/runs_road_damage/exp4_1050ti_fix/weights/best.pt")

    # 2. เริ่มการเทรน Fine-tuning รอบใหม่ (exp6)
    results = model.train(
        data="C:/Project-test/road-damage-detection.v5i.yolov8/data.yaml",  # Dataset v5i (ที่มี Hard Negatives/Backgrounds)
        epochs=30,             # เทรน 30 Epochs
        imgsz=640,
        batch=4,               # คุม VRAM สำหรับ 1050 Ti
        workers=0,             # ป้องกัน Memory leak บน Windows
        device=0,
        patience=15,
        amp=False,             # GTX 1050 Ti ปิดไว้เสถียรที่สุด
        
        # --- Hyperparameters สำหรับ Fine-tuning เพื่อรักษาความเก่งเดิม ---
        lr0=0.001,             # ลด Learning Rate ลง 10 เท่า ไม่ให้ทำลายน้ำหนักความรู้เดิม
        lrf=0.01,              # Final Learning Rate สัมพันธ์กับ lr0
        warmup_epochs=1.0,     # วอร์มอัปสั้นๆ เพียง 1 Epoch เพราะโมเดลเก่งอยู่แล้ว
        freeze=10,             # แช่แข็ง Backbone 10 เลเยอร์แรก ล็อกการจำลักษณะแผลเดิมไว้
        
        # --- ป้องกัน CUDA Crash และ Stagnation ---
        deterministic=True,    # เปิดไว้ป้องกัน CUDA error: an illegal instruction บน GTX 1050 Ti
        
        project="C:/Project-test/runs_road_damage",
        name="exp6_finetune_v5i"
    )
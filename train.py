import os
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from ultralytics import YOLO

if __name__ == '__main__':
    torch.cuda.empty_cache()

    # โหลด best.pt จาก exp6 มาตั้งต้น
    model = YOLO("C:/Project-test/runs_road_damage/exp6_finetune_v5i/weights/best.pt")

    # สั่ง train ใหม่โดยส่งค่า config ทั้งหมดเข้าไป (ไม่ต้องใช้ resume=True)
    results = model.train(
        data="C:/Project-test/road-damage-detection.v5i.yolov8/data.yaml",
        epochs=60,             # เทรนต่ออีก 20 Epochs
        imgsz=640,
        batch=4,
        workers=0,
        device=0,
        patience=15,
        amp=False,
        lr0=0.0005,            # ลด LR ลงนิดหน่อยสำหรับการเก็บรายละเอียดช่วงท้าย
        deterministic=True,
        project="C:/Project-test/runs_road_damage",
        name="exp7_finetune_ext"
    )
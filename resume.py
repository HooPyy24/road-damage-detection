from ultralytics import YOLO

if __name__ == '__main__':
    # ชี้ไปที่ last.pt ของ exp7
    model = YOLO("C:/Project-test/runs_road_damage/exp7_finetune_ext/weights/last.pt")

    # สั่ง Resume ระบบจะอ่าน epochs: 60 จาก args.yaml แล้วรันต่อจาก  ถึง 60 ทันที
    results = model.train(resume=True)
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. โหลดไฟล์ weights ที่ดีที่สุดจากการเทรน
    model = YOLO("C:/Project-test/runs_road_damage/exp7_finetune_ext/weights/best.pt")

    # ==========================================
    # กรณีที่ 1: สั่ง Predict รูปภาพ (ทั้งโฟลเดอร์ หรือ รูปเดียว)
    # ==========================================
    image_results = model.predict(
        source="C:/Project-test/road-damage-detection.v5i.yolov8/test/test_pothole",  # พาธโฟลเดอร์รูป หรือไฟล์รูปภาพ (.jpg/.png)
        conf=0.25,             # Threshold ความมั่นใจขั้นต่ำ (25%)
        save=True,             # บันทึกรูปภาพที่มีกรอบ Bounding Box
        save_txt=True,         # (Optional) บันทึกพิกัดกรอบเป็นไฟล์ .txt
        project="C:/Project-test/runs_road_damage/predictions",
        name="predict_images"
    )

    # ==========================================
    # กรณีที่ 2: สั่ง Predict ไฟล์วิดีโอ
    # ==========================================
    video_results = model.predict(
        source="C:/Project-test/test_video.mp4",  # พาธไฟล์วิดีโอ (.mp4/.avi)
        conf=0.25,             # Threshold ความมั่นใจขั้นต่ำ
        save=True,             # บันทึกผลลัพธ์วิดีโอที่มี Bounding Box (.avi/.mp4)
        project="C:/Project-test/runs_road_damage/predictions",
        name="predict_video"
    )
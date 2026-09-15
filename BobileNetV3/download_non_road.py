import os
import urllib.request

train_non_road = r"C:\Project-test\BobileNetV3\dataset\train\non_road"
val_non_road   = r"C:\Project-test\BobileNetV3\dataset\val\non_road"

os.makedirs(train_non_road, exist_ok=True)
os.makedirs(val_non_road, exist_ok=True)

TOTAL_IMAGES = 2000
TRAIN_COUNT  = 1600  # 80% ของ 2,000

print(f"กำลังเริ่มดาวน์โหลดภาพ non_road ขนาด HD จำนวน {TOTAL_IMAGES} รูป...")

success_count = 0
for i in range(TOTAL_IMAGES):
    url = f"https://picsum.photos/400/400?random={i}"
    
    # 80% แรกเข้า train ส่วนที่เหลือเข้า val
    save_dir = train_non_road if i < TRAIN_COUNT else val_non_road
    save_path = os.path.join(save_dir, f"hd_non_road_{i}.jpg")
    
    try:
        urllib.request.urlretrieve(url, save_path)
        success_count += 1
        # พิมพ์บอกสถานะทุกๆ 50 รูป
        if (i + 1) % 50 == 0:
            print(f"โหลดสำเร็จแล้ว {i + 1}/{TOTAL_IMAGES} รูป...")
    except Exception as e:
        print(f"รูปที่ {i} โหลดไม่สำเร็จ ข้ามไปรูปถัดไป...")

print("\nดาวน์โหลดภาพครบเรียบร้อย!")
print(f"- Train (non_road): {len(os.listdir(train_non_road))} รูป")
print(f"- Val (non_road): {len(os.listdir(val_non_road))} รูป")
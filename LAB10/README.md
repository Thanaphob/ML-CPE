# ML-10 Object Detection II

โปรเจกต์ตรวจจับวัตถุด้วย **YOLOv8** (Ultralytics) เพื่อตรวจจับ **คน (person)** และ **ยานพาหนะ (vehicle)** ในภาพ วิดีโอ และกล้องเว็บแคมแบบเรียลไทม์ ครอบคลุมการเตรียมข้อมูล การฝึกสอนหลาย configuration การประเมินผล และการทดสอบการตรวจจับ

## 1. ภาพรวมโปรเจกต์

- Fine-tune โมเดล `yolov8n.pt` (pretrained) กับ dataset คนและยานพาหนะ
- เปรียบเทียบผลเมื่อใช้ **จำนวน epoch**, **ขนาดภาพ (imgsz)** และ **batch size** ต่างกัน
- ประเมินด้วย Precision, Recall, mAP50 และ mAP50-95 บนชุดทดสอบ (test)
- ทดสอบตรวจจับจากภาพนิ่งและกล้องเว็บแคมแบบเรียลไทม์

## 2. Dataset

ใช้ [YOLO Person Vehicle Detection Dataset Annotated](https://www.kaggle.com/datasets/haseebhsb/yolo-person-vehicle-detection-dataset-annotated)

| รายการ | รายละเอียด |
|---|---|
| Class | `0: person`, `1: vehicle` |
| Train | 1,600 ภาพ |
| Validation | 200 ภาพ (หลังแบ่งส่วนหนึ่งเป็น test) |
| Test | 200 ภาพ |

dataset ต้นฉบับมีเฉพาะชุด train และ val จึงใช้ `prepare_dataset.py` ย้ายครึ่งหนึ่งของ val (พร้อมไฟล์ label) ไปเป็นชุด test แล้วสร้าง `dataset/data.yaml` ใหม่

รูปแบบไฟล์ label ต่อหนึ่งวัตถุ (ค่าพิกัดเป็นสัดส่วน 0-1 เทียบกับขนาดภาพ):

```text
class_id x_center y_center width height
```

## 3. โครงสร้างโปรเจกต์

```text
ML-10-Object-Detection-II/
├── dataset/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
├── runs/
├── prepare_dataset.py
├── train.py
├── train_all.py
├── evaluate.py
├── image_test.py
├── realtime_test.py
├── requirements.txt
└── README.md
```

| ไฟล์ | หน้าที่ |
|---|---|
| `prepare_dataset.py` | แบ่งชุด test จาก val และสร้าง `data.yaml` |
| `train.py` | ฝึกโมเดลหนึ่ง configuration |
| `train_all.py` | ฝึกหลาย configuration ในครั้งเดียว ประเมินบนชุด test และสรุปผลเป็นตาราง `runs/summary.csv` |
| `evaluate.py` | ประเมินโมเดลบนชุด test |
| `image_test.py` | สุ่มภาพจากชุด test มาตรวจจับและแสดงผล |
| `realtime_test.py` | ตรวจจับแบบเรียลไทม์จากกล้องเว็บแคม |

## 4. การติดตั้ง

```bash
pip install -r requirements.txt
```

สคริปต์เลือกอุปกรณ์เองอัตโนมัติ ใช้ GPU (CUDA) ถ้ามี ไม่เช่นนั้นใช้ CPU การฝึกในโปรเจกต์นี้ทำบน CPU

## 5. วิธีใช้งาน

**1) เตรียมข้อมูล** วางโฟลเดอร์ `images` และ `labels` ของ dataset ไว้ใน `dataset/` แล้วรัน

```bash
python prepare_dataset.py
```

**2) ฝึกโมเดลทุก configuration**

```bash
python train_all.py
```

ผลของแต่ละ configuration อยู่ใน `runs/<ชื่อ config>/` (น้ำหนักโมเดล `weights/best.pt`, `results.csv`, กราฟ loss และ mAP) ผลบนชุด test อยู่ใน `runs/<ชื่อ config>_test/` และตารางสรุปอยู่ที่ `runs/summary.csv`

หากต้องการข้ามการฝึกของ configuration ที่ฝึกเสร็จแล้ว ให้ใส่ `"skip_train": True` ในรายการ `CONFIGS` ของ `train_all.py` สคริปต์จะใช้ `best.pt` เดิมมาประเมินและรวมในตารางสรุป

**3) ทดสอบตรวจจับ** แก้ Path `weights` ใน `image_test.py` และ `realtime_test.py` ให้ชี้ไปที่ `best.pt` ของ configuration ที่เลือก (ในตัวอย่างใช้ `runs/e10_img480_b16/weights/best.pt`) แล้วรัน

```bash
python image_test.py
python realtime_test.py
```

- `image_test.py` กดปุ่มใดก็ได้เพื่อไปภาพถัดไป กด `q` เพื่อออก
- `realtime_test.py` เปิดกล้องหมายเลข 0 กด `q` เพื่อออก

## 6. Configuration ที่ทดลอง

ทุก configuration ใช้โมเดล `yolov8n.pt` เริ่มฝึกจาก pretrained weights ใหม่ทุกครั้ง เปลี่ยนทีละตัวแปรจาก configuration ฐาน `e10_img480_b8`

| ชื่อ | epochs | imgsz | batch | สิ่งที่เปลี่ยนจากฐาน |
|---|---|---|---|---|
| `e5_img480_b8` | 5 | 480 | 8 | จำนวน epoch |
| `e10_img480_b8` | 10 | 480 | 8 | ฐานเปรียบเทียบ |
| `e10_img320_b8` | 10 | 320 | 8 | ขนาดภาพ |
| `e10_img480_b16` | 10 | 480 | 16 | batch size |

## 7. ผลการทดลอง

ผลบนชุด test (200 ภาพ) จาก `runs/summary.csv`

| Configuration | เวลาฝึก (นาที) | Train loss | Val loss | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `e5_img480_b8` | 16.8 | 3.240 | 3.246 | 0.868 | 0.756 | 0.846 | 0.606 |
| `e10_img480_b8` | 29.8 | 2.892 | 3.100 | 0.882 | 0.781 | 0.876 | 0.641 |
| `e10_img320_b8` | 13.0 | 2.936 | 3.026 | 0.863 | 0.755 | 0.861 | 0.604 |
| `e10_img480_b16` | 32.0 | 2.824 | 3.132 | 0.838 | 0.802 | 0.875 | 0.644 |

หมายเหตุ Train loss และ Val loss คือผลรวมของ box loss, cls loss และ dfl loss ใน epoch สุดท้าย

### การวิเคราะห์

- **จำนวน epoch (5 เทียบกับ 10):** การเพิ่มเป็น 10 epoch ทำให้ mAP50 เพิ่มขึ้นประมาณ 0.03 และ mAP50-95 เพิ่มขึ้นประมาณ 0.035 ค่า loss ลดลง แต่ใช้เวลาฝึกเพิ่มเกือบเท่าตัว
- **ขนาดภาพ (480 เทียบกับ 320):** ภาพขนาดเล็กลงทำให้ความแม่นยำลดลงเล็กน้อย (mAP50-95 ลดลงประมาณ 0.037 และ Recall ลดลง) แต่ฝึกเร็วขึ้นประมาณ 2.3 เท่า
- **Batch size (8 เทียบกับ 16):** ผลใกล้เคียงกันมาก mAP50 เท่าเดิม ส่วน mAP50-95 ต่างกันประมาณ 0.003 ซึ่งน้อยเกินกว่าจะสรุปว่า batch 16 ดีกว่า Recall สูงขึ้นขณะที่ Precision ลดลง
- **Train loss เทียบกับ Val loss:** ที่ 10 epoch Val loss สูงกว่า Train loss เล็กน้อย ยังไม่พบสัญญาณ overfitting ที่ชัดเจน
- **Configuration ที่ mAP50-95 สูงสุด:** `e10_img480_b16` (0.644) แต่ต่างจาก `e10_img480_b8` (0.641) เพียงเล็กน้อย

## 8. ข้อจำกัด

- แต่ละ configuration ฝึกเพียงครั้งเดียว ความต่างของค่าที่น้อยกว่าประมาณ 0.01 อาจเป็นความผันผวนตามธรรมชาติ ไม่ควรตีความว่าเป็นความต่างที่แท้จริง
- ชุด test มี 200 ภาพ ซึ่งค่อนข้างน้อย ค่าที่ได้จึงมีความไม่แน่นอน
- ฝึกเพียง 5-10 epoch บน CPU เนื่องจากข้อจำกัดด้านเวลา การฝึกนานขึ้นหรือใช้โมเดลที่ใหญ่ขึ้นมีโอกาสได้ผลดีกว่านี้
- ยังไม่ได้วัดความเร็วในการตรวจจับ (FPS) อย่างเป็นระบบ

## 9. แนวทางพัฒนาต่อ

- เพิ่มจำนวน epoch และทดลอง `imgsz=640`
- ทดลองโมเดลที่ใหญ่ขึ้น เช่น `yolov8s.pt`
- วิเคราะห์ความผิดพลาดแยกตาม class (person และ vehicle)
- วัด FPS ของการตรวจจับแบบเรียลไทม์
- ฝึกซ้ำหลายครั้งด้วย random seed ต่างกันเพื่อดูความผันผวนของผล

## 10. เอกสารอ้างอิง

- [YOLO Person Vehicle Detection Dataset Annotated (Kaggle)](https://www.kaggle.com/datasets/haseebhsb/yolo-person-vehicle-detection-dataset-annotated)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
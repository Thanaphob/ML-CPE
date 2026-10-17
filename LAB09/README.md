# Lab 9 - Object Detection I: R-CNN และ Fast R-CNN

## ภาพรวม

Lab นี้ทดลองสร้างโมเดลตรวจจับวัตถุ 2 แบบที่อิงแนวคิดจาก R-CNN และ Fast R-CNN โดยเป็นเวอร์ชันย่อแบบง่ายสำหรับการศึกษา (educational simplified baseline) ไม่ใช่การ implement สถาปัตยกรรมต้นฉบับทั้งหมด

ทั้งสองแบบทำงานเป็น **ตัวจำแนกภาพที่ถูกตัดมาจากกล่องจริง (crop classification)** ไม่ใช่ตัวตรวจจับวัตถุที่หาตำแหน่งเองได้แบบสมบูรณ์:

- ตัดภาพตามกล่อง ground truth จาก label แบบ YOLO ออกมาทีละวัตถุ
- ปรับขนาดเป็น 64x64 แล้วส่งเข้า CNN เพื่อทายว่าเป็น class อะไร
- ไม่มีการทำนายตำแหน่งกล่อง (bounding box regression) และไม่มี region proposal ระหว่างฝึก

| | 01-R-CNN | 02-Fast-R-CNN |
|---|---|---|
| โครงสร้าง | Sequential, Conv 32→64→128 | Functional API, Conv 32→64→128→256 + BatchNorm |
| ก่อนชั้นจำแนก | GlobalAveragePooling2D | MaxPooling2D(pool_size=4) จำลอง ROI pooling แบบคงที่ |
| Data augmentation | ไม่มี | RandomFlip, RandomRotation, RandomZoom |

## Dataset

ใช้ dataset รูปแบบ YOLO (เช่น [YOLO Person Vehicle Detection Dataset](https://www.kaggle.com/datasets/haseebhsb/yolo-person-vehicle-detection-dataset-annotated)) ที่มี `data.yaml` ระบุรายชื่อ class และโฟลเดอร์ภาพ/label แยกตาม train/val/test

โครงสร้างที่โค้ดรองรับ (ตรวจหาอัตโนมัติ):
```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

ถ้าไม่มี split `test` โค้ดจะใช้ `val` แทนโดยอัตโนมัติ และแจ้งเตือนในเทอร์มินัล

## โครงสร้างโปรเจกต์

```
LAB09/
├── 01-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
│       ├── model.keras
│       ├── classes.json
│       ├── evaluation.json
│       └── image_result.jpg
├── 02-Fast-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
│       ├── model.keras
│       ├── classes.json
│       ├── evaluation.json
│       └── image_result.jpg
└── dataset/
    ├── data.yaml
    ├── images/
    └── labels/
```

`classes.json` ถูกสร้างตอนเทรน เก็บรายชื่อ class ที่อ่านมาจาก `data.yaml` และถูกใช้ร่วมกันโดย evaluate / image_test / realtime_test

## การติดตั้ง

```bash
pip install -r requirements.txt
```

## วิธีใช้งาน

รันตามลำดับในแต่ละโฟลเดอร์ (`01-R-CNN` หรือ `02-Fast-R-CNN`):

```bash
cd 02-Fast-R-CNN

# 1. เทรนโมเดล
python train.py --epochs 20

# 2. ประเมินผลบน test set
python evaluate.py --split test

# 3. ทดสอบกับภาพเดี่ยว (สุ่มจาก test set ถ้าไม่ระบุ --image)
python image_test.py

# 4. ทดสอบแบบเรียลไทม์ผ่านกล้อง
python realtime_test.py
```

พารามิเตอร์ที่ปรับได้:
- `--limit` จำนวนภาพต้นทางที่ใช้ (0 = ทั้งหมด)
- `--epochs` จำนวนรอบการเทรน (มี EarlyStopping ตรวจ `val_loss` คอยหยุดก่อนถ้าไม่ดีขึ้น)
- `--batch` ขนาด batch

## ผลการทดลอง (2 class: person / vehicle, test set)

| Metric | 01-R-CNN | 02-Fast-R-CNN |
|---|---|---|
| Accuracy | 0.92 | 0.760 |
| Precision (macro) | 0.92 | 0.753 |
| Recall (macro) | 0.91 | 0.758 |
| F1 (macro) | 0.91 | 0.754 |

ผลของ Fast R-CNN: เทรน 20 epoch (หยุดอัตโนมัติที่ epoch 6 ด้วย EarlyStopping, ใช้ weight จาก epoch 2 ที่ val_loss ดีที่สุด)

| Class | Precision | Recall | F1 | จำนวนตัวอย่าง |
|---|---|---|---|---|
| person | 0.81 | 0.77 | 0.79 | 467 |
| vehicle | 0.69 | 0.75 | 0.72 | 324 |

ผลถือว่าดีเมื่อเทียบกับการเดาสุ่ม (~50% สำหรับ 2 class) โมเดลจำแนก person ได้แม่นยำกว่า vehicle เล็กน้อย ซึ่งสอดคล้องกับจำนวนตัวอย่างในชุดข้อมูลที่ person มีมากกว่า

## ข้อจำกัด

- เป็นตัวจำแนกภาพที่ถูกตัดมาแล้ว ไม่ใช่ตัวตรวจจับวัตถุที่หาตำแหน่งได้เอง ต่างจาก Lab 10 (YOLO) ที่ทำนายทั้งตำแหน่งและ class พร้อมกัน
- ไม่มีการคำนวณ mAP เพราะไม่ได้ทำนาย bounding box
- `realtime_test.py` ใช้วิธี sliding window สุ่มตำแหน่งแบบคงที่ (ไม่ใช่ region proposal network จริง) จึงช้าและแม่นยำน้อยกว่าการตรวจจับแบบ end-to-end
- ฝึกบนกล่องจริงเท่านั้น โมเดลไม่เคยเห็นตัวอย่างพื้นหลัง (background) ระหว่างเทรน

## งานในอนาคต

- เพิ่ม region proposal ที่แท้จริง (เช่น Selective Search) แทนการสุ่ม sliding window
- เพิ่มการทำนาย bounding box regression เพื่อคำนวณ mAP ได้
- เปรียบเทียบความเร็วและความแม่นยำกับ YOLO ใน Lab 10 อย่างเป็นระบบ

## อ้างอิง

- Girshick, R. et al. "Rich feature hierarchies for accurate object detection and semantic segmentation" (R-CNN), 2014
- Girshick, R. "Fast R-CNN", 2015
- [YOLO Person Vehicle Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/haseebhsb/yolo-person-vehicle-detection-dataset-annotated?select=yolo-person-vehicle-dataset)
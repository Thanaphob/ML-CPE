# LAB07 - Convolutional Neural Network (CNN)

โปรเจกต์นี้สร้าง CNN ด้วย Python สำหรับการจำแนกรูปภาพ (image recognition) ครอบคลุมตั้งแต่การโหลดภาพ, การ preprocess, การแบ่งชุดข้อมูล, การเทรนโมเดล CNN, การประเมินผล, ไปจนถึงการทำนาย

## ข้อมูล (Data)

Intel Image Classification: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

มีทั้งหมด 6 คลาส: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`

ดาวน์โหลด dataset จาก Kaggle แล้วนำโฟลเดอร์ `seg_train` และ `seg_test` มาวางไว้ใน `data/` (ส่วนโฟลเดอร์ `seg_pred` ไม่ได้ใช้ในโปรเจกต์นี้ ไม่ต้องเก็บไว้ก็ได้):

```text
LAB07/
├── classification/
├── data/
│   ├── seg_train/
│   │   └── seg_train/
│   │       ├── buildings/
│   │       ├── forest/
│   │       ├── glacier/
│   │       ├── mountain/
│   │       ├── sea/
│   │       └── street/
│   └── seg_test/
│       └── seg_test/
│           ├── buildings/
│           ├── forest/
│           ├── glacier/
│           ├── mountain/
│           ├── sea/
│           └── street/
```

## โครงสร้างโปรเจกต์

```text
LAB07/
│
├── data/                        
│   ├── seg_train/
│   └── seg_test/
│
├── classification/
│   ├── main.py                  # ไฟล์หลักของ pipeline การเทรน
│   ├── data_loader.py           # โหลดภาพและข้ามไฟล์ที่เสีย
│   ├── preprocessing.py         # ปรับขนาดภาพและแปลง BGR เป็น RGB
│   ├── split_data.py            # แบ่งข้อมูล training เป็น train/validation
│   ├── cnn_model.py             # สร้าง เทรน บันทึก และทำนายผลด้วยโมเดล CNN
│   ├── evaluate.py              # คำนวณ accuracy, classification report, confusion matrix, กราฟการเทรน
│   ├── test_cnn.py              # ทดสอบโมเดลที่เทรนแล้วด้วยภาพตัวอย่างแบบสุ่ม
│   └── outputs/                
│       ├── features.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       ├── y_test.npy
│       ├── cnn_model.keras
│       ├── history.json
│       ├── confusion_matrix.png
│       ├── training_history.png
│       └── prediction_sample.png
└── requirements.txt
```

## ติดตั้ง (Setup)

```bash
pip install -r requirements.txt
```

## วิธีใช้งาน (Usage)

เทรนโมเดล (รันจากภายในโฟลเดอร์ `classification/`):

```bash
python main.py
```

ขั้นตอนนี้จะโหลด dataset, preprocess ภาพ, แบ่งข้อมูล training เป็น train/validation, เทรน CNN, ประเมินผลบนชุด test, และบันทึกผลลัพธ์ทั้งหมดไว้ที่ `outputs/`

ทดสอบโมเดลที่เทรนแล้วด้วยภาพตัวอย่างแบบสุ่ม:

```bash
python test_cnn.py
```

สร้างกราฟ training history ใหม่โดยไม่ต้องเทรนซ้ำ (อ่านข้อมูลจาก `outputs/history.json`):

```bash
python replot.py
```

## สรุปโปรเจกต์

โปรเจกต์นี้ใช้ CNN เพื่อจำแนกภาพทิวทัศน์ออกเป็น 6 ประเภท ภาพจะถูกโหลดจากโฟลเดอร์ dataset โดยอัตโนมัติ ปรับขนาดให้เท่ากัน และแปลงจาก BGR เป็น RGB ระหว่างขั้นตอน preprocessing จากนั้นข้อมูล training จะถูกแบ่งออกเป็นชุด train และ validation ก่อนนำไปเทรนโมเดล CNN ส่วนชุด test ที่มีมาให้จาก dataset จะถูกใช้สำหรับการประเมินผลสุดท้าย โมเดลที่เทรนแล้วจะถูกประเมินด้วยค่า accuracy, precision, recall, F1-score, confusion matrix และกราฟ training history เพื่อดูประสิทธิภาพในการจำแนกภาพ

## โมเดล

CNN ประกอบด้วย 4 conv block (32 → 64 → 128 → 256 filters) แต่ละ block ตามด้วย batch normalization และ max pooling ก่อนเข้าสู่ dense layer สำหรับการจำแนกพร้อม dropout มีการทำ data augmentation (random flip, rotation, zoom) ระหว่างการเทรนเพื่อลด overfitting และใช้ early stopping กับการลด learning rate อัตโนมัติเมื่อผลลัพธ์ไม่ดีขึ้น

ค่า hyperparameter หลัก (อยู่ใน `main.py`):

| พารามิเตอร์ | ค่า |
|---|---|
| `IMG_SIZE` | 64 |
| `MAX_PER_CLASS` | 1500 |
| `EPOCHS` | 15 |
| `BATCH_SIZE` | 64 |
| `VAL_SIZE` | 0.1 |
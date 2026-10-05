# ML-08 Deep Convolutional Neural Network (DCNN): การจำแนกภาพผัก 15 ชนิด

โปรเจกต์นี้สร้าง DCNN สถาปัตยกรรมแบบ VGG ด้วย Python และ Keras เพื่อจำแนกภาพผัก 15 ชนิด ครอบคลุมการโหลดภาพ, การเตรียมข้อมูล, การเทรนโมเดล, การประเมินผล และการเปรียบเทียบ DCNN หลายแบบกับจำนวน epochs ที่ต่างกัน

## ข้อมูล (Dataset)

- แหล่งที่มา: [Vegetable Image Dataset (Kaggle)](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- ภาพสีของผัก 15 คลาส ขนาดต้นฉบับ 224×224 พิกเซล
- แบ่งเป็น `train` / `validation` / `test` มาให้แล้วในชุดข้อมูล (ชุด test มี 3,000 ภาพ คลาสละ 200 ภาพ)
- คลาสทั้ง 15: Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato
- รายชื่อคลาสที่โปรแกรมตรวจพบจริงถูกบันทึกใน `classification/outputs/classes.json`

โฟลเดอร์ข้อมูลไม่ได้อัปโหลดขึ้น git เพราะมีขนาดใหญ่ (ตั้งค่าไว้ใน `.gitignore`) ผู้ที่ต้องการรันโปรเจกต์ต้องดาวน์โหลดเอง:

1. ดาวน์โหลดชุดข้อมูลจากลิงก์ Kaggle ด้านบนแล้วแตกไฟล์
2. วางไว้ในโฟลเดอร์ `data/` ให้ภายในมี `train`, `validation`, `test` (จะซ้อนอยู่ในโฟลเดอร์ย่อยอีกชั้นก็ได้ เช่น `data/Vegetable Images/train` โปรแกรมค้นหาให้เอง)
3. ค่าเริ่มต้นของ `DATA_PATH` ใน `main.py` และ `experiments.py` คือ `../data` (โฟลเดอร์ `data` อยู่ระดับเดียวกับ `classification`) ถ้าวางไว้ที่อื่นให้แก้ค่า `DATA_PATH`

## โครงสร้างโปรเจกต์

```text
LAB08/
├── data/                           # ชุดข้อมูลจาก Kaggle (ไม่อยู่ใน git)
│   └── Vegetable Images/
│       ├── train/
│       ├── validation/
│       └── test/
│
└── classification/
    ├── main.py                     # ขั้นตอนหลักในการเทรนโมเดล
    ├── data_loader.py              # โหลดภาพจาก train/validation/test และข้ามไฟล์ที่เสีย
    ├── preprocessing.py            # ปรับขนาดภาพ และแปลง BGR เป็น RGB
    ├── vgg_model.py                # สร้าง เทรน บันทึก และทำนายด้วยโมเดล VGG
    ├── evaluate.py                 # Accuracy, classification report, confusion matrix, กราฟการเทรน
    ├── test_vgg.py                 # ทดสอบโมเดลด้วยภาพสุ่มจากชุด test
    ├── experiments.py              # เปรียบเทียบ DCNN หลายแบบและจำนวน epochs
    │
    └── outputs/                    # ไฟล์ที่สร้างขึ้นหลังรัน
        ├── classes.json
        ├── X_val.npy
        ├── X_test.npy
        ├── y_train.npy
        ├── y_val.npy
        ├── y_test.npy
        ├── vgg_model.keras
        ├── history.json
        ├── confusion_matrix.png
        ├── training_history.png
        ├── prediction_sample.png
        │
        └── experiments/            # ผลจาก experiments.py
            ├── results.csv
            ├── accuracy_curves.png
            ├── checkpoint_comparison.png
            ├── history_<ชื่อโมเดล>.json
            └── model_<ชื่อโมเดล>.keras
```

## การติดตั้งและการรัน

ต้องใช้ Python 3.10 ขึ้นไป ติดตั้งไลบรารีอย่างใดอย่างหนึ่ง (โปรแกรมเลือก backend ให้เอง: ถ้ามี `torch` จะใช้ torch ถ้าไม่มีจะใช้ tensorflow)

```bash
pip install keras torch opencv-python scikit-learn matplotlib numpy
```

หรือ

```bash
pip install tensorflow opencv-python scikit-learn matplotlib numpy
```

จากนั้นรันตามลำดับ

```bash
cd classification
python main.py
python test_vgg.py
python experiments.py
```

## ขั้นตอนของ main.py

1. โหลดภาพจาก `train`, `validation`, `test` และปรับขนาดเป็น 48×48
2. แปลงข้อมูลเป็นอาร์เรย์ (uint8) พร้อมใช้งาน
3. บันทึกข้อมูลที่ใช้ทดสอบลง `outputs/`
4. เทรนโมเดล VGG
5. ทำนายบนชุด test
6. ประเมินผลและบันทึกกราฟ

## โมเดล

โมเดลหลักใน `vgg_model.py` เป็น VGG-style CNN สร้างเอง

- เรียงบล็อก Conv 3×3 + Batch Normalization แล้ว Max Pooling ตามรูปแบบ VGG
- `VGG_SMALL` (ค่าเริ่มต้นใน `main.py`) ใช้ 10 ชั้น Conv และ `VGG16` ใช้ 13 ชั้น Conv ตามโครงสร้าง VGG-16 (หัวท้ายต่างจาก VGG-16 มาตรฐาน)
- ใช้ Global Average Pooling ตามด้วย Dense 512 → 256 พร้อม Dropout 0.5 และชั้นผลลัพธ์แบบ softmax
- ปรับสเกลภาพเป็น 0–1 ภายในโมเดล และมี Data Augmentation (พลิกซ้ายขวา, หมุน, ซูม) ช่วงเทรนเท่านั้น
- Adam (learning rate 3e-4), EarlyStopping และ ReduceLROnPlateau โดยดูค่า `val_accuracy`

ค่าตั้งต้นใน `main.py`: `IMG_SIZE = 64`, `EPOCHS = 12`, `BATCH_SIZE = 128`, `MAX_PER_CLASS = 500`

ผลของโมเดลหลักดูได้จากบรรทัด `Accuracy:` และ classification report ที่ `main.py` พิมพ์ตอนจบ รวมถึง `outputs/confusion_matrix.png` และ `outputs/training_history.png`

## การเปรียบเทียบโมเดลและจำนวน epochs (experiments.py)

เทรนแต่ละโมเดลครั้งเดียวจนครบ 10 epochs แล้ววัดความแม่นยำที่ epoch 3, 6 และ 10 ซึ่งเทียบเท่ากับการเทรนด้วยจำนวน epochs นั้นโดยตรง เพราะใช้ learning rate คงที่และไม่มี early stopping

**ส่วน A: โมเดลที่สร้างเอง** (ภาพ 48×48, Adam 1e-3, batch 64) เปรียบเทียบจำนวนชั้น Conv และจำนวนนิวรอน

| โมเดล | ชั้น Conv | Dense |
|---|---|---|
| Small | 4 | 64 |
| Medium | 7 | 256 |
| Large | 10 | 512-256 |

**ส่วน B: โมเดลสำเร็จรูปที่ใช้น้ำหนัก ImageNet** (ภาพ 128×128) ล็อกตัวโมเดลไว้และเทรนเฉพาะชั้นจำแนกที่ต่อท้าย ได้แก่ VGG16, MobileNetV2 และ EfficientNetB0

### ผลการทดลอง (Test Accuracy, %)

| โมเดล | พารามิเตอร์ทั้งหมด | พารามิเตอร์ที่เทรน | 3 epochs | 6 epochs | 10 epochs | เวลา (วินาที) |
|---|---|---|---|---|---|---|
| Small (4 Conv, Dense 64) | 0.41M | 0.41M | 77.87 | 81.60 | 95.67 | 224 |
| Medium (7 Conv, Dense 256) | 0.65M | 0.65M | 83.63 | 88.10 | **96.90** | 432 |
| Large (10 Conv, Dense 512-256) | 2.18M | 2.18M | 80.80 | 89.80 | 92.93 | 551 |
| VGG16 (pretrained) | 14.72M | 0.008M | 98.23 | 99.23 | 99.50 | 594 |
| MobileNetV2 (pretrained) | 2.28M | 0.019M | 99.43 | 99.63 | 99.70 | **173** |
| EfficientNetB0 (pretrained) | 4.07M | 0.019M | 99.53 | 99.67 | **99.80** | 195 |

ตัวเลขทั้งหมดมาจาก `outputs/experiments/results.csv` เวลาคือเวลาเทรนรวม 10 epochs และของโมเดลสำเร็จรูปรวมช่วงดึงฟีเจอร์จากภาพไว้ด้วย กราฟเปรียบเทียบอยู่ที่ `outputs/experiments/accuracy_curves.png` และ `outputs/experiments/checkpoint_comparison.png`

### สรุปและวิเคราะห์ผล

1. **ผลของจำนวน epochs**: ความแม่นยำเพิ่มขึ้นตามจำนวน epochs ในทุกโมเดล โมเดลที่สร้างเองได้ประโยชน์มากที่สุด เช่น Small เพิ่มจาก 77.87% เป็น 95.67% (+17.8 จุด) เมื่อเทรนจาก 3 เป็น 10 epochs ส่วนโมเดลสำเร็จรูปแทบอิ่มตัวตั้งแต่ epoch 3 (เพิ่มเพียง 0.3–1.3 จุด)
2. **ผลของจำนวนชั้น Conv และนิวรอน**: ในกลุ่มที่สร้างเอง Medium (7 Conv, Dense 256) ให้ผลดีที่สุดที่ 10 epochs (96.90%) ส่วน Large ซึ่งลึกที่สุดได้ต่ำสุด (92.93%) โดยค่า train accuracy ของ Large ที่ 10 epochs ก็ยังไม่สูง (95.39%) จึงสันนิษฐานว่าโมเดลที่ลึกกว่าต้องใช้จำนวน epochs มากกว่าจึงจะเรียนรู้ได้เต็มที่ ซึ่งยังไม่ได้ทดลองยืนยัน ดังนั้นการเพิ่มชั้น Conv ไม่ได้ทำให้ผลดีขึ้นเสมอไปเมื่อจำกัดจำนวน epochs
3. **เทียบกับโมเดลสำเร็จรูป**: โมเดลที่ใช้น้ำหนัก ImageNet ได้ 99.5–99.8% สูงกว่าโมเดลที่สร้างเองทั้งหมด ทั้งที่เทรนเพียงชั้นจำแนกเล็ก ๆ (ไม่เกิน 19,215 พารามิเตอร์) เพราะตัวโมเดลเรียนรู้ลักษณะของภาพมาจาก ImageNet แล้ว
4. **ความคุ้มค่าด้านเวลา**: MobileNetV2 ใช้เวลาน้อยที่สุด (173 วินาที) แต่ความแม่นยำ (99.70%) ใกล้เคียงโมเดลที่ดีที่สุด ส่วน VGG16 มีพารามิเตอร์มากที่สุดและใช้เวลานานที่สุดในกลุ่มสำเร็จรูป

### ข้อจำกัดของการเปรียบเทียบ

- ชุด test มี 3,000 ภาพ ความต่าง 0.1–0.3 จุดระหว่างโมเดลสำเร็จรูป (ประมาณ 3–9 ภาพ) น้อยเกินกว่าจะสรุปว่าโมเดลใดดีกว่ากัน และแต่ละโมเดลเทรนเพียงครั้งเดียว
- กลุ่มที่สร้างเองและกลุ่มสำเร็จรูปไม่ได้เทียบในเงื่อนไขเท่ากัน: ขนาดภาพต่างกัน (48×48 กับ 128×128) และกลุ่มสำเร็จรูปล็อกตัวโมเดลไว้
- ค่า train accuracy คำนวณระหว่างเทรนที่เปิด Dropout และ Data Augmentation จึงอาจต่ำกว่า validation accuracy ในบางแถว

## ไฟล์ผลลัพธ์

ดูผลได้จากไฟล์ใน `classification/outputs/`

- `confusion_matrix.png`: ตารางความสับสนของแต่ละคลาสบนชุด test
- `training_history.png`: กราฟ accuracy และ loss ของชุด train และ validation
- `prediction_sample.png`: ตัวอย่างการทำนายภาพสุ่มจากชุด test
- `experiments/results.csv`: ตารางเปรียบเทียบ accuracy ทุกโมเดลที่ epoch 3, 6, 10
- `experiments/accuracy_curves.png` และ `experiments/checkpoint_comparison.png`: กราฟเปรียบเทียบ

## อ้างอิง

- ชุดข้อมูล: Vegetable Image Dataset, Kaggle — https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset
- Simonyan, K. และ Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG) — https://arxiv.org/abs/1409.1556
- Sandler, M. และคณะ (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks — https://arxiv.org/abs/1801.04381
- Tan, M. และ Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks — https://arxiv.org/abs/1905.11946
- Keras Applications (โมเดลสำเร็จรูปพร้อมน้ำหนัก ImageNet) — https://keras.io/api/applications/
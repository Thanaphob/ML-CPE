# Gesture-controlled Presentation

ระบบควบคุมการนำเสนอสไลด์ (PowerPoint / Canva / Google Slides / PDF) ด้วยท่าทางมือผ่านกล้องเว็บแคม แทนการกดคีย์บอร์ดหรือเมาส์ โดยใช้เทคนิค Computer Vision (Hand Landmark Detection) ร่วมกับโมเดล Machine Learning ที่เทรนเอง (Gesture Classification) เพื่อแปลงท่ามือเป็นคำสั่งเปลี่ยนสไลด์แบบ real-time

---

## แนวคิดโปรเจกต์

ระบบทำงานโดยรับภาพจากกล้องแบบต่อเนื่อง (real-time) แล้วผ่านขั้นตอนดังนี้:

1. **อ่านภาพจากกล้อง** ด้วย OpenCV ทีละเฟรม
2. **ตรวจจับตำแหน่งมือ** ด้วย MediaPipe Hands (pretrained model) ซึ่งจะคืนค่าตำแหน่งจุดสำคัญบนมือ (landmark) ทั้งหมด 21 จุดต่อมือ 1 ข้าง
3. **สกัด Feature** จากตำแหน่ง landmark โดย normalize ให้ไม่ขึ้นกับตำแหน่ง/ระยะห่างจากกล้อง (ดูรายละเอียดใน `landmark_utils.py`)
4. **จำแนกท่าทาง (Gesture Classification)** ด้วยโมเดล **SVM ที่เทรนเอง** จากข้อมูล landmark ที่เก็บจากท่ามือจริง
5. **สั่งคำสั่งคีย์บอร์ด** ผ่าน PyAutoGUI เพื่อจำลองการกดปุ่มลูกศรซ้าย/ขวา ทำให้โปรแกรมนำเสนอสไลด์ที่เปิดอยู่ (PowerPoint, Canva, Google Slides, PDF) เปลี่ยนหน้าให้จริง

### ท่าทางที่รองรับ

| ท่ามือ | คำสั่ง |
|---|---|
| ชูนิ้วชี้อย่างเดียว | Next Slide (กด →) |
| ชูนิ้วชี้ + นิ้วกลาง | Previous Slide (กด ←) |
| แบมือ / กำมือ / ท่าอื่นๆ | Idle (ไม่ทำอะไร) |

มีระบบ **cooldown** (ค่าเริ่มต้น 1.2 วินาที) กันไม่ให้สั่งเปลี่ยนสไลด์ถี่เกินไปจากการจับมือที่สั่นหรือไม่นิ่ง

---

## โครงสร้างไฟล์

```
.
├── main.py                 ไฟล์หลักที่รัน: เปิดกล้อง, เรียกใช้โมดูลอื่น, แสดงผลบนจอ
├── config.py                ค่าตั้งต้นทั้งหมด (cooldown, dry run, เลขกล้อง, ความมั่นใจของโมเดล)
├── hand_detector.py         ครอบการทำงานของ MediaPipe Hands (ตรวจจับ + วาด landmark)
├── landmark_utils.py        แปลง landmark ดิบ -> feature vector ที่ normalize แล้ว (ใช้ร่วมกันทั้ง 3 ไฟล์ด้านล่าง)
├── collect_data.py          สคริปต์เก็บข้อมูล landmark พร้อม label ลง data/gesture_data.csv
├── train_model.py           เทรนโมเดล SVM จากข้อมูลที่เก็บ + ประเมินผล + บันทึกเป็น gesture_model.pkl
├── gesture_classifier.py    โหลดโมเดลที่เทรนไว้มาทำนายท่าทาง (มี rule-based fallback ถ้ายังไม่ได้เทรน)
├── slide_controller.py      รับคำสั่งแล้วกดคีย์บอร์ดจริงผ่าน PyAutoGUI พร้อม cooldown
├── requirements.txt         รายชื่อไลบรารีที่ต้องติดตั้ง
├── data/gesture_data.csv    ข้อมูลที่เก็บจาก collect_data.py (สร้างขึ้นเองตอนใช้งาน)
└── gesture_model.pkl         โมเดลที่เทรนเสร็จแล้ว (สร้างขึ้นเองตอนรัน train_model.py)
```

---

## สิ่งที่ต้องใช้ (Requirements)

### Hardware
- คอมพิวเตอร์ที่มีเว็บแคม (ในตัวเครื่องหรือต่อ USB แยกก็ได้)

### Software
- **Python 3.11** (สำคัญ: ปัจจุบัน MediaPipe ยังไม่รองรับ Python เวอร์ชันใหม่มากๆ เช่น 3.12–3.14 อย่างสมบูรณ์ แนะนำให้ใช้ 3.11 เพื่อความเข้ากันได้)
- ไลบรารีตามไฟล์ `requirements.txt`:
  - `opencv-python` — เปิดกล้องและประมวลผลภาพ
  - `mediapipe==0.10.21` — ตรวจจับตำแหน่งมือ (hand landmark detection)
  - `pyautogui` — จำลองการกดคีย์บอร์ด
  - `numpy`, `pandas` — จัดการข้อมูลตัวเลข/ตาราง
  - `scikit-learn` — เทรนและประเมินผลโมเดล SVM
  - `joblib` — บันทึก/โหลดโมเดลที่เทรนแล้ว

---

## วิธีติดตั้งและใช้งาน

### 1. ติดตั้ง environment

```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. เก็บข้อมูลสำหรับเทรนโมเดล

```bash
python collect_data.py
```

- กด `0` / `1` / `2` เพื่อเลือกโหมดเก็บท่า **IDLE / NEXT / PREV**
- กด `r` เพื่อเริ่ม/หยุดบันทึก แล้วค้างท่านั้นไว้หน้ากล้องประมาณ 10-15 วินาทีต่อท่า (ได้ข้อมูลหลายร้อยแถวต่อท่าโดยอัตโนมัติ ไม่ต้องถ่ายทีละรูป)
- ทำซ้ำจนครบทั้ง 3 ท่า แล้วกด `q` เพื่อบันทึกลง `data/gesture_data.csv`

### 3. เทรนโมเดล

```bash
python train_model.py
```

สคริปต์จะแบ่งข้อมูล train/test, เทรนโมเดล SVM, พิมพ์ค่า accuracy และ confusion matrix ออกมาให้ดู แล้วบันทึกโมเดลลงไฟล์ `gesture_model.pkl`

### 4. รันระบบใช้งานจริง

เปิดไฟล์สไลด์ (PowerPoint / Canva / Google Slides / PDF) เข้าสู่โหมด Presentation แบบเต็มจอค้างไว้ และคลิกให้หน้าต่างนั้น active อยู่ จากนั้นเปิด terminal อีกหน้าต่างแล้วรัน:

```bash
python main.py
```

ยกมือทำท่าตามตารางด้านบนเพื่อเปลี่ยนสไลด์ กด `q` ที่หน้าต่างกล้องเพื่อออกจากโปรแกรม

> ถ้ายังไม่ได้ทำขั้นตอนที่ 2-3 (ยังไม่มีไฟล์ `gesture_model.pkl`) โปรแกรมจะสลับไปใช้โหมด rule-based (เทียบ pattern จำนวนนิ้วตรงๆ) โดยอัตโนมัติ เพื่อให้ยังใช้งานได้ระหว่างที่ยังไม่ได้เทรนโมเดล

### ปรับแต่งค่าได้ที่ `config.py`
- `COOLDOWN_SECONDS` — เวลาขั้นต่ำระหว่างการสั่งเปลี่ยนสไลด์แต่ละครั้ง
- `DRY_RUN` — ตั้งเป็น `True` เพื่อทดสอบระบบโดยไม่ส่งคำสั่งคีย์บอร์ดจริง
- `CAM_INDEX` — เลขกล้อง กรณีมีกล้องหลายตัวในเครื่อง

---

## แนวคิดเชิง Machine Learning ที่เกี่ยวข้อง

| ขั้นตอน ML | อยู่ที่ไฟล์ | รายละเอียด |
|---|---|---|
| Data Collection | `collect_data.py` | เก็บตำแหน่ง landmark 21 จุดของมือ พร้อม label ของแต่ละท่า |
| Preprocessing / Feature Extraction | `landmark_utils.py` | normalize ตำแหน่ง landmark ให้ไม่ขึ้นกับตำแหน่งมือในเฟรมและระยะห่างจากกล้อง |
| Training | `train_model.py` | เทรนโมเดล **SVM (Support Vector Machine)** ด้วย `scikit-learn` |
| Evaluation | `train_model.py` | วัด accuracy, classification report, confusion matrix จาก test set (แบ่ง 80/20) |
| Deployment / Inference | `gesture_classifier.py` | โหลดโมเดลที่เทรนเสร็จมาทำนายท่าทางแบบ real-time |

- **Hand Landmark Detection**: MediaPipe Hands ใช้โมเดล Deep Learning (CNN) ที่เทรนไว้ล่วงหน้าโดย Google เพื่อตรวจจับตำแหน่งข้อต่อและปลายนิ้วบนมือจากภาพ — ส่วนนี้เป็น pretrained model ไม่ได้เทรนเอง
- **Gesture Classifier (SVM)**: ส่วนที่เทรนเองในโปรเจกต์นี้ รับ feature vector (ตำแหน่ง landmark ที่ normalize แล้ว) เป็น input แล้วจำแนกว่าเป็นท่า IDLE / NEXT / PREV

---

## แหล่งอ้างอิง (References)

- MediaPipe Hands — Google AI Edge. [https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- MediaPipe GitHub Repository. [https://github.com/google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)
- OpenCV Documentation. [https://docs.opencv.org/](https://docs.opencv.org/)
- PyAutoGUI Documentation. [https://pyautogui.readthedocs.io/](https://pyautogui.readthedocs.io/)
- scikit-learn: Support Vector Machines. [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)
- Python 3.11 Release. [https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)

---
         LAB 05: Support Vector Machine (SVM) Classification
                   Dataset: Adult Income Dataset

[1] PROJECT STRUCTURE

```text
ML-05-SVM/
├── data_adult_income/
│   └── adult.csv           # ไฟล์ชุดข้อมูล CSV จาก Kaggle/UCI
├── classification/
│   ├── main.py             # สคริปต์หลักสำหรับรัน Pipeline ทั้งหมด
│   ├── data_loader.py      # โหลดไฟล์ CSV และลบค่าสูญหาย (Missing values '?')
│   ├── preprocessing.py    # ทำ One-Hot Encoding และ Feature Scaling (StandardScaler)
│   ├── split_data.py       # แบ่งข้อมูลเป็น Train (80%) และ Test (20%)
│   ├── svm_model.py        # สร้างและเทรนแบบจำลอง SVM Classifier (Kernel RBF)
│   ├── evaluate.py         # ประเมินผลโมเดล (Accuracy, Classification Report, Confusion Matrix)
│   └── outputs/            # โฟลเดอร์เก็บผลลัพธ์อัตโนมัติหลังรันเสร็จ
│       ├── scaler.pkl      # เซฟตัวปรับสเกลข้อมูล (StandardScaler)
│       ├── svm_model.pkl   # เซฟโมเดล SVM ที่เทรนเสร็จแล้ว
│       ├── classes.json    # บันทึก Mapping ของ Class Target
│       └── confusion_matrix.png # รูปภาพกราฟแสดง Confusion Matrix
├── requirements.txt        # รายชื่อ Python Libraries ที่จำเป็น
├── link-data.txt           # ลิงก์ที่มาของ Dataset
└── README.txt              # อธิบายรายละเอียดโปรเจกต์และการใช้งาน


[2] REQUIREMENTS & INSTALLATION
------------------------------------------------------------------------
ไลบรารีที่จำเป็นต้องใช้ในโปรเจกต์นี้:
- Python 3
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib

คำสั่งสำหรับติดตั้งไลบรารีทั้งหมด:
pip install -r requirements.txt


[3] HOW TO RUN
------------------------------------------------------------------------
1. ดาวน์โหลดไฟล์ `adult.csv` และนำไปวางไว้ในโฟลเดอร์ `data_adult_income/`
2. เปิด Terminal ใน VS Code แล้วเข้าไปที่โฟลเดอร์ `classification`:

   cd classification

3. รันสคริปต์หลักด้วยคำสั่ง:

   python main.py


[4] MODEL PERFORMANCE & RESULTS
------------------------------------------------------------------------
ผลการทดสอบแบบจำลอง SVM (Kernel = RBF, C = 1.0):

- Total Samples (Cleaned) : 45,222 แถว (ลบ missing values '?' ออก 3,620 แถว)
- Train Set / Test Set    : 36,177 / 9,045 แถว
- Encoded Features         : 96 คอลัมน์ (หลังทำ One-Hot Encoding)
- Overall Accuracy        : ~84.26%

* หมายเหตุ: โมเดลมีประสิทธิภาพสูงมากในการทำนายกลุ่ม `<=50K` (F1-score 0.90) 
  และมี Recall ประมาณ 0.57 สำหรับกลุ่ม `>50K` เนื่องจากสัดส่วนข้อมูลตั้งต้น
  มีความเอนเอียง (Class Imbalance)
========================================================================
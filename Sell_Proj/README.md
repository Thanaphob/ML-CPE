# SSH Brute Force Detection with SVM

ตรวจจับการโจมตี SSH แบบ Brute Force / Password Spraying ด้วย Support Vector Machine (SVM)

## โครงสร้างโปรเจกต์

```
ssh_bruteforce_svm/
├── data/
│   └── ssh_anomaly_dataset.csv   # raw event log (timestamp, ip, username, status, label)
├── outputs/
│   └── confusion_matrix.png      # ผลการประเมินโมเดล
├── data_loader.py                 # โหลด raw log
├── preprocessing.py               # แปลง raw log -> feature vector ต่อ (IP, ช่วงเวลา)
├── split_data.py                  # แบ่ง train/test + scale
├── svm_model.py                   # นิยาม/เทรน SVM
├── evaluate.py                    # ประเมินผล + confusion matrix
├── main.py                        # รันทั้ง pipeline
├── requirements.txt
└── README.md
```

## Feature ที่ใช้

จาก raw event log ของแต่ละ IP ในหน้าต่างเวลา 1 นาที คำนวณ:

| Feature | ความหมาย |
|---|---|
| `failed_attempts` | จำนวนครั้งที่ login ผิดในหน้าต่างเวลา |
| `avg_time_between` | เวลาเฉลี่ยระหว่างแต่ละครั้งที่พยายาม (วินาที) |
| `unique_usernames` | จำนวน username ที่ต่างกันซึ่งถูกลอง |
| `success_ratio` | สัดส่วนที่ login สำเร็จ |

Label: `0 = normal`, `1 = attack` (รวม brute_force และ brute_force_connection_issue)

## วิธีรัน

```bash
pip install -r requirements.txt
python main.py
```

จะได้ accuracy, classification report ใน terminal และ confusion matrix ที่ `outputs/confusion_matrix.png`

## หมายเหตุ

- ตัด label `config_anomaly` ออก (มีแค่ 2 แถว และเป็นปัญหา config เซิร์ฟเวอร์ ไม่ใช่พฤติกรรม login)
- ปรับขนาดหน้าต่างเวลาได้ที่ตัวแปร `WINDOW` ใน `preprocessing.py`

* **ข้อมูลอ้างอิง**
SSH Anomaly Dataset
Kaggle : https://www.kaggle.com/datasets/mdwiraputradananjaya/ssh-anomaly-dataset

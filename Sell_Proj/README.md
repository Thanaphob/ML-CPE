# Network Port Scan Detection using Support Vector Machine (SVM)

ระบบตรวจจับพฤติกรรมการแอบสแกนพอร์ตในระบบเครือข่ายคอมพิวเตอร์ (Network Port Scan Detection) โดยใช้ขั้นตอนวิธี Support Vector Machine (SVM) ร่วมกับเคอร์เนล Radial Basis Function (RBF) เพื่อทำหน้าที่เป็นด่านหน้าในการคัดกรองภัยคุกคามก่อนที่ผู้โจมตีจะค้นพบช่องโหว่ของเซิร์ฟเวอร์

---

## 1. ที่มาและความสำคัญ (Problem Statement)
* **ปัญหา:** ก่อนที่ผู้ไม่หวังดีจะเริ่มโจมตีระบบ มักจะต้องทำการสำรวจ (Reconnaissance) ด้วยการใช้เครื่องมืออย่าง Nmap เพื่อสแกนหาพอร์ตที่เปิดอยู่ (Port Scanning) 
* **ผลกระทบ:** หากไม่สามารถตรวจจับขั้นตอนนี้ได้ ผู้โจมตีจะมีข้อมูลบริการที่รันอยู่และเลือกเจาะระบบผ่านช่องโหว่ได้อย่างแม่นยำ
* **การแก้ปัญหาด้วย SVM:** การใช้ Rule-based ดั้งเดิมอาจตั้งค่าได้ช้าและรับมือกับเทคนิคการสแกนแบบสุ่มได้ยาก การนำ SVM เข้ามาช่วยวิเคราะห์พฤติกรรมแพ็กเก็ตช่วยให้แยกระหว่าง Traffic ทั่วไปกับ Traffic การสแกนได้อย่างรวดเร็วและใช้พลังงานประมวลผลต่ำ

---

## 2. ชุดข้อมูลที่ใช้ (Dataset)
* **Dataset:** NSL-KDD (`KDDTrain+_20Percent.txt`) ซึ่งเป็นชุดข้อมูลมาตรฐานสำหรับทดสอบระบบ Intrusion Detection System (IDS)
* **การแบ่งกลุ่มเป้าหมาย (Binary Classification):**
  * `0: Normal` — ทราฟฟิกการใช้งานปกติของผู้ใช้
  * `1: Port Scan` — กลุ่มการโจมตีประเภท Probe/Reconnaissance ได้แก่ `portsweep`, `ipsweep`, `satan` และ `nmap`

---

## 3. ฟีเจอร์ที่คัดเลือกมาใช้งาน (Feature Selection)
คัดเลือกเฉพาะ 10 ฟีเจอร์ตัวเลขที่สะท้อนถึงพฤติกรรมการส่งสัญญาณตรวจหาพอร์ต:
1. `duration`: ระยะเวลาในการเชื่อมต่อ
2. `src_bytes`: จำนวนไบต์ที่ส่งจากต้นทาง
3. `dst_bytes`: จำนวนไบต์ที่ส่งกลับจากปลายทาง
4. `count`: จำนวนการเชื่อมต่อไปยังโฮสต์เดียวกันในกรอบเวลาสั้นๆ
5. `srv_count`: จำนวนการเชื่อมต่อไปยังบริการ (Service) เดียวกัน
6. `diff_srv_rate`: อัตราส่วนการส่งไปยังหลายบริการที่แตกต่างกัน (พฤติกรรมหลักของการสแกนพอร์ต)
7. `dst_host_count`: สถิติการเข้าถึงโฮสต์ปลายทาง
8. `dst_host_srv_count`: สถิติการเข้าถึงพอร์ตบริการปลายทาง
9. `dst_host_same_srv_rate`: อัตราส่วนการเรียกใช้บริการเดิม
10. `dst_host_diff_srv_rate`: อัตราส่วนการเรียกบริการหลากหลายพอร์ตในระดับปลายทาง

---

## 4. สถาปัตยกรรมและขั้นตอนการทำงาน (System Architecture & Pipeline)

```text
[ Raw Dataset: NSL-KDD ]
          │
          ▼
[ data_loader.py ] ───────> กำหนด Header Columns ทั้ง 43 คอลัมน์ และโหลดเข้า DataFrame
          │
          ▼
[ preprocessing.py ] ─────> กรองเฉพาะคลาส Normal และ Port Scan (Probe) พร้อมคัดเลือก 10 ฟีเจอร์
          │
          ▼
[ split_data.py ] ────────> แบ่งชุดข้อมูล Train 70% และ Test 30% (ใช้ Stratified Sampling)
          │
          ▼
[ Feature Scaling ] ──────> ทำ StandardScaler เพื่อปรับ Normalization ให้เวกเตอร์ระยะทางใน SVM เที่ยงตรง
          │
          ▼
[ svm_model.py ] ─────────> เทรนโมเดล SVC(kernel='rbf', C=1.0) เพื่อคำนวณหาระนาบ Hyperplane
          │
          ▼
[ evaluate.py ] ──────────> ประเมินผลลัพธ์ (Accuracy, Precision, Recall, Confusion Matrix)
```

---

## 5. ผลการทดลองและการวัดผล (Experimental Results)
จากการทดสอบกับ Test Set จำนวน **4,722 ตัวอย่าง**:

| Metric | Normal (0) | Port Scan (1) | Overall |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.96 | 0.90 | - |
| **Recall** | 0.99 | 0.76 | - |
| **F1-Score** | 0.97 | 0.82 | - |
| **Accuracy** | - | - | **95%** |

### Confusion Matrix Breakdown:
* **True Negative (Normal $\to$ Normal):** 3,980 เคส
* **True Positive (Port Scan $\to$ Port Scan):** 519 เคส
* **False Positive (Normal ผิดเป็น Port Scan):** 55 เคส (ต่ำมาก ลดปัญหาการแจ้งเตือนรบกวนผู้ดูแลระบบ)
* **False Negative (Port Scan หลุดรอด):** 168 เคส

---

## 6. จุดเด่นในการนำไปประยุกต์ใช้งานจริง (System Selling Points)
* **Lightweight & High Efficiency:** โมเดล SVM กินทรัพยากร CPU และ RAM ต่ำกว่า Deep Learning อย่างเห็นได้ชัด สามารถนำไปติดตั้งบนอุปกรณ์เครือข่ายระดับขอบเขต (Edge Gateway / Router / Firewall) ได้โดยตรง
* **Low False Alarm Rate:** ค่า Precision ฝั่งทราฟฟิกปกติสูงถึง 96% และมี False Positive เพียง 55 เคส ช่วยให้ระบบไม่บล็อก IP ของผู้ใช้งานทั่วไปโดยพลการ
* **Early-stage Prevention:** สกัดกั้นตั้งแต่ระยะสแกนพอร์ต ช่วยยับยั้งแผนการโจมตีก่อนที่ผู้ไม่หวังดีจะเริ่มส่ง Exploit เข้ามาสู่ระบบเครือข่าย

---

## 7. วิธีการรันโปรเจกต์ (Usage)

```bash
# ติดตั้ง Library ที่จำเป็น
pip install -r requirement.txt

# รันโปรแกรมหลัก
python main.py
```
* **ข้อมูลอ้างอิง**
NSL-KDD Network Security, Information Security, Cyber Security
Kaggle : https://www.kaggle.com/datasets/hassan06/nslkdd
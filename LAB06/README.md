# Adult Income Classification using Neural Network (MLP)

โปรเจกต์นี้เป็นการปรับปรุงและพัฒนาระบบ Machine Learning จากเดิมที่เป็นโครงสร้างจำแนกรูปภาพ ให้กลายมาเป็น **Multi-Layer Perceptron (MLP)** สำหรับจำแนกข้อมูลตารางโดยใช้ชุดข้อมูล **Adult Income Dataset** เพื่อทำนายว่าบุคคลนั้นมีรายได้มากกว่า $50K ต่อปีหรือไม่ (`>50K` หรือ `<=50K`)

---

```text
ML-06-NN/
│
├── dataset/
│   └── adult.csv               # ไฟล์ชุดข้อมูลดิบ Adult Income
│
├── classification/
│   ├── main.py                 # Pipeline หลักสำหรับควบคุมการทำงานทั้งหมด
│   ├── data_loader.py          # โหลด CSV, จัดการ Missing Values, One-Hot Encoding, Scaling
│   ├── split_data.py           # แบ่งข้อมูล Train / Validation / Test
│   ├── nn_model.py             # สถาปัตยกรรม MLP, ฟังก์ชัน Train และ Predict
│   ├── evaluate.py             # ประเมินประสิทธิภาพ (Accuracy, Report, Confusion Matrix)
│   ├── test_nn.py              # สุ่มตัวอย่างข้อมูลทดสอบและพิมพ์ผลการทำนายใน Terminal
│   └── outputs/                # โฟลเดอร์เก็บ Model Weights, Metrics และรูปภาพกราฟ
│       ├── features.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy / X_val.npy / X_test.npy
│       ├── y_train.npy / y_val.npy / y_test.npy
│       ├── nn_model.keras
│       ├── history.json
│       ├── confusion_matrix.png
│       └── training_history.png
│
└── requirements.txt            # รายการไลบรารีที่จำเป็น

## ขั้นตอนและกระบวนการพัฒนา
    
    ###1. การจัดการและเตรียมข้อมูล (Data Pipeline Adaptation)
    
        **การเปลี่ยน Data Loader: แทนที่คำสั่งประมวลผลรูปภาพ (cv2, imread, การย่อขนาดภาพ) ด้วยการใช้ pandas.read_csv**
        
        **การจัดการ Missing Values: ค้นหาเครื่องหมาย ? ในชุดข้อมูลและตัดแถวที่มีค่าว่างออก (dropna())**
        
        **Categorical Encoding: แปลงตัวแปรกลุ่ม เช่น workclass, education, occupation ให้อยู่ในรูปตัวเลขด้วยวิธี One-Hot Encoding (pd.get_dummies(drop_first=True)) ได้ฟีเจอร์ทั้งหมดรวม 95 ฟีเจอร์**
        
        **Feature Scaling: ทำการแปลงข้อมูลตัวเลขทั้งหมดด้วย StandardScaler เพื่อปรับค่าให้อยู่ในรูป Zero-mean ($Mean = 0, Std = 1$) เหมาะสำหรับการฝึกสอน Neural Network**

    ###2. การแบ่งชุดข้อมูล (Dataset Splitting)
        แบ่งข้อมูลออกเป็น 3 ชุด เพื่อใช้ในการเทรน ปรับจูน และวัดผลจริง:

        Training Set (70%): 4,200 ตัวอย่าง

        Validation Set (10%): 600 ตัวอย่าง

        Test Set (20%): 1,200 ตัวอย่าง

        กำหนด stratify=y เพื่อรักษาสัดส่วนการกระจายตัวของคลาส <=50K และ >50K ให้เท่ากันในทุกชุดข้อมูล

    ###3. การปรับสถาปัตยกรรมโมเดล (Model Architecture Refactoring)
            ตัดเลเยอร์เฉพาะของรูปภาพออก: นำ layers.Rescaling(1/255) และ layers.Flatten() ออก เนื่องจากข้อมูลนำเข้าเป็น 1D Feature Vector อยู่แล้ว

            โครงสร้างเลเยอร์ Fully-Connected (MLP):

            Input Layer: รับขนาด 95 Features

            Dense Layer 1: 256 Nodes (ReLU) + BatchNormalization + Dropout(0.4)

            Dense Layer 2: 128 Nodes (ReLU) + BatchNormalization + Dropout(0.4)

            Dense Layer 3: 64 Nodes (ReLU) + Dropout(0.3)

            Output Layer: 1 Node (Sigmoid Activation สำหรับ Binary Classification)

            Optimization & Callbacks:

            Optimizer: Adam(learning_rate=0.001)

            Loss Function: Binary Crossentropy

            EarlyStopping: ตรวจจับ val_loss เพื่อหยุดการเทรนอัตโนมัติเมื่อโมเดลเริ่ม Overfit (Patience = 5)

            ReduceLROnPlateau: ปรับลด Learning Rate อัตโนมัติเมื่อ Loss เริ่มไม่ลดลง

    ###4. การปรับการทดสอบโมเดล (Inference Refactoring)
        ในไฟล์ test_nn.py ได้เปลี่ยนจากการพล็อตภาพ Grid 2x2 ด้วย ax.imshow() มาเป็นการสุ่มหยิบตัวอย่างตารางขึ้นมาทำนายผล และพิมพ์ตารางสรุป Predicted, True, Confidence Score

    ทดสอบผลลัพธ์
        pip install tensorflow pandas scikit-learn matplotlib numpy #ติดตั้ง Library ที่ต้องใช้

        cd classification
        python main.py
        #สั่งรัน Pipeline การเทรนและการประเมินผล

        python test_nn.py #สั่งรันการทดสอบสุ่มทำนายข้อมูล

        ผลการทดลองและประเมินผล (Evaluation Results)
            Overall Test Accuracy: 81.17%

            Classification Report:

            Class <=50K: Precision = 0.84, Recall = 0.78, F1-Score = 0.80

            Class >50K: Precision = 0.79, Recall = 0.85, F1-Score = 0.82

            Training History & Convergence: โมเดลตัดรอบการเทรนที่ Epoch 19 จากระบบ Early Stopping โดยค่า Train Loss อยู่ที่ 0.3787 และ Validation Loss อยู่ที่ 0.3967 แสดงว่าโมเดลไม่มีปัญหา Overfitting

* **ข้อมูลอ้างอิง**
Adult income dataset
Kaggle : https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
ที่มาของ Code ที่ใช้อ้างอิง : https://github.com/aproot-en/Machine-Learning-Course.git
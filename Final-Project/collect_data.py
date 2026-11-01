import csv
import os

import cv2

import config
from hand_detector import HandDetector
from landmark_utils import extract_features, feature_names

OUTPUT_DIR = "data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "gesture_data.csv")

LABELS = {ord("0"): "IDLE", ord("1"): "NEXT", ord("2"): "PREV"}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(config.CAM_INDEX)
    if not cap.isOpened():
        print("เปิดกล้องไม่ได้ ลองเปลี่ยนค่า CAM_INDEX ใน config.py")
        return

    detector = HandDetector()

    rows = []
    counts = {"IDLE": 0, "NEXT": 0, "PREV": 0}
    current_label = "IDLE"
    recording = False

    print("เริ่มเก็บข้อมูลแล้ว")
    print("กด 0/1/2 เพื่อเลือกท่า (IDLE/NEXT/PREV), กด 'r' เพื่อเริ่ม-หยุดบันทึก, 'q' เพื่อบันทึกไฟล์และออก")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            results = detector.process(frame)

            hand_found = False
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                detector.draw_landmarks(frame, hand_landmarks)
                hand_found = True

                if recording:
                    features = extract_features(hand_landmarks)
                    rows.append(features + [current_label])
                    counts[current_label] += 1

            # overlay แสดงสถานะ
            rec_text = "RECORDING" if recording else "paused"
            rec_color = (0, 0, 255) if recording else (200, 200, 200)

            cv2.rectangle(frame, (0, 0), (640, 90), (0, 0, 0), -1)
            cv2.putText(frame, f"Label: {current_label}   [{rec_text}]", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
            cv2.putText(frame,
                        f"IDLE: {counts['IDLE']}   NEXT: {counts['NEXT']}   PREV: {counts['PREV']}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "0/1/2 = select label | r = record on/off | q = save & quit",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            if not hand_found:
                cv2.putText(frame, "No hand detected", (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Data Collection (press 'q' to save & quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                recording = not recording
            elif key in LABELS:
                current_label = LABELS[key]
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    if not rows:
        print("ไม่มีข้อมูลที่บันทึกไว้ ไม่ได้สร้างไฟล์ CSV")
        return

    file_exists = os.path.isfile(OUTPUT_PATH)
    with open(OUTPUT_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(feature_names() + ["label"])
        writer.writerows(rows)

    print(f"บันทึกข้อมูล {len(rows)} แถว ลงไฟล์ {OUTPUT_PATH} แล้ว")
    print(f"สรุปจำนวนต่อท่า: {counts}")


if __name__ == "__main__":
    main()

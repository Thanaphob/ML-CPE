from collections import deque

import cv2

import config
from hand_detector import HandDetector
from gesture_classifier import classify_gesture, preload_model
from slide_controller import SlideController


def draw_overlay(frame, status_text, last_label):
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Status: {status_text}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Last command: {last_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    # โหลด + warm-up โมเดล ML ล่วงหน้าตรงนี้ (ก่อนเปิดกล้อง/เข้าลูปวิดีโอ)
    # กันไม่ให้ไปโหลดตอนเฟรมแรกที่เจอมือ ซึ่งจะทำให้ภาพค้าง/กระตุกตอนยกมือ
    print("กำลังโหลดโมเดล...")
    preload_model()

    cap = cv2.VideoCapture(config.CAM_INDEX)
    if not cap.isOpened():
        print("เปิดกล้องไม่ได้ ลองเปลี่ยนค่า CAM_INDEX ใน config.py เป็น 1 หรือ 2 แล้วรันใหม่")
        return

    detector = HandDetector()
    controller = SlideController()
    recent_actions = deque(maxlen=config.STABLE_FRAMES_REQUIRED)

    print("เริ่มทำงานแล้ว — กด 'q' ที่หน้าต่างกล้องเพื่อออก")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            results = detector.process(frame)

            status_text = "No hand detected"

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness_label = results.multi_handedness[0].classification[0].label

                detector.draw_landmarks(frame, hand_landmarks)

                action, label, confidence = classify_gesture(hand_landmarks, handedness_label)
                status_text = label

                if confidence is not None and confidence < config.MIN_CONFIDENCE:
                    action = "IDLE"

                recent_actions.append(action)

                is_stable = (
                    len(recent_actions) == recent_actions.maxlen
                    and all(a == action for a in recent_actions)
                )

                if is_stable:
                    controller.trigger(action, label)
                else:
                    status_text += "  (stabilizing...)"

            draw_overlay(frame, status_text, controller.last_label)
            cv2.imshow("Gesture Presentation Control (press 'q' to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
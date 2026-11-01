import os

from landmark_utils import extract_features

MODEL_PATH = "gesture_model.pkl"

# ปลายนิ้วแต่ละนิ้ว (landmark id) ตามที่ MediaPipe กำหนด — ใช้เฉพาะตอน rule-based fallback
TIP_IDS = [4, 8, 12, 16, 20]

LABEL_TO_ACTION = {
    "NEXT": ("NEXT", "Next Slide -> (ML)"),
    "PREV": ("PREV", "<- Previous Slide (ML)"),
    "IDLE": ("IDLE", "Idle (ML)"),
}

_model_bundle = None
_model_checked = False


def _load_model():

    global _model_bundle, _model_checked

    if _model_checked:
        return _model_bundle

    _model_checked = True
    if os.path.isfile(MODEL_PATH):
        import joblib  # import ตรงนี้ เผื่อเครื่องที่ยังไม่ได้ติดตั้ง sklearn/joblib แต่ไม่ได้ใช้โหมด ML
        _model_bundle = joblib.load(MODEL_PATH)
        print(f"[gesture_classifier] โหลดโมเดลที่เทรนไว้จาก {MODEL_PATH} สำเร็จ (ML mode)")
    else:
        print(f"[gesture_classifier] ไม่พบ {MODEL_PATH} — ใช้ rule-based fallback แทน "
              f"(รัน collect_data.py แล้ว train_model.py เพื่อเปิดใช้ ML mode)")

    return _model_bundle


def _count_fingers_rule_based(hand_landmarks, handedness_label):

    landmarks = hand_landmarks.landmark
    fingers = []

    if handedness_label == "Right":
        fingers.append(1 if landmarks[TIP_IDS[0]].x < landmarks[TIP_IDS[0] - 1].x else 0)
    else:
        fingers.append(1 if landmarks[TIP_IDS[0]].x > landmarks[TIP_IDS[0] - 1].x else 0)

    for tip_id in TIP_IDS[1:]:
        fingers.append(1 if landmarks[tip_id].y < landmarks[tip_id - 2].y else 0)

    return fingers


def _classify_rule_based(hand_landmarks, handedness_label):
    fingers = _count_fingers_rule_based(hand_landmarks, handedness_label)

    if fingers == [0, 1, 0, 0, 0]:
        return "NEXT", "Next Slide -> (rule)"
    if fingers == [0, 1, 1, 0, 0]:
        return "PREV", "<- Previous Slide (rule)"
    if sum(fingers) == 5:
        return "IDLE", "Open Palm (rule)"
    if sum(fingers) == 0:
        return "IDLE", "Fist (rule)"
    return "IDLE", f"{sum(fingers)} fingers (rule, unmapped)"


def classify_gesture(hand_landmarks, handedness_label):

    bundle = _load_model()

    if bundle is not None:
        features = extract_features(hand_landmarks)
        predicted_label = bundle["model"].predict([features])[0]
        return LABEL_TO_ACTION.get(predicted_label, ("IDLE", "Unknown (ML)"))

    return _classify_rule_based(hand_landmarks, handedness_label)
TIP_IDS = [4, 8, 12, 16, 20]


def count_fingers(hand_landmarks, handedness_label):

    landmarks = hand_landmarks.landmark
    fingers = []

    if handedness_label == "Right":
        fingers.append(1 if landmarks[TIP_IDS[0]].x < landmarks[TIP_IDS[0] - 1].x else 0)
    else:
        fingers.append(1 if landmarks[TIP_IDS[0]].x > landmarks[TIP_IDS[0] - 1].x else 0)

    for tip_id in TIP_IDS[1:]:
        fingers.append(1 if landmarks[tip_id].y < landmarks[tip_id - 2].y else 0)

    return fingers


def classify_gesture(fingers):

    total = sum(fingers)

    if fingers == [0, 1, 0, 0, 0]:
        return "NEXT", "Next Slide ->"

    if fingers == [0, 1, 1, 0, 0]:
        return "PREV", "<- Previous Slide"

    if total == 5:
        return "IDLE", "Open Palm (idle)"

    if total == 0:
        return "IDLE", "Fist (idle)"

    return "IDLE", f"{total} fingers (unmapped)"

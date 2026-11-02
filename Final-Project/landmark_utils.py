import numpy as np

WRIST_ID = 0
MIDDLE_FINGER_MCP_ID = 9


def extract_features(hand_landmarks):

    points = np.array(
        [[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32
    )  # shape (21, 2)

    wrist = points[WRIST_ID]
    relative = points - wrist

    scale = np.linalg.norm(relative[MIDDLE_FINGER_MCP_ID])
    if scale < 1e-6:
        scale = 1e-6

    normalized = relative / scale

    return normalized.flatten().tolist()


def feature_names():
    
    names = []
    for i in range(21):
        names.append(f"x{i}")
        names.append(f"y{i}")
    return names

import time

import pyautogui

import config


class SlideController:
    def __init__(self):
        self.last_action_time = 0.0
        self.last_label = "..."

    def trigger(self, action, label):
        
        if action not in ("NEXT", "PREV"):
            return False

        now = time.time()
        if (now - self.last_action_time) <= config.COOLDOWN_SECONDS:
            return False

        if not config.DRY_RUN:
            pyautogui.press("right" if action == "NEXT" else "left")

        self.last_action_time = now
        self.last_label = label
        return True

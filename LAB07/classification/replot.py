import json
from evaluate import plot_history

class FakeHistory:
    def __init__(self, history_dict):
        self.history = history_dict

with open("outputs/history.json") as f:
    history_dict = json.load(f)

plot_history(FakeHistory(history_dict), "outputs/training_history.png")
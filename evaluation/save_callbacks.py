import csv
import logging
import pickle
from collections import deque, Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environments.recommendation_env import ReturnStateTuple
from util.helpers import flatten

logger = logging.getLogger(__name__)


class PredictionSaver:
    def __init__(self, log_path, file_mode):
        self.log_path = log_path
        self.buffer = deque(maxlen=1000)
        self.file_mode = file_mode
        self.counter = Counter()

    def add_predictions(self, predictions: np.ndarray):
        self.counter.update(predictions.flatten())
        if predictions.ndim == 2:
            self.buffer.extend(predictions)
        else:
            self.buffer.append(predictions)

    def save(self):
        if self.log_path is None:
            return
        ar = np.array(self.buffer)
        logger.info("Minimal action: {}, max action: {}, unique items: {} in {} items".format(ar.min(), ar.max(),
                                                                                              len(np.unique(ar)),
                                                                                              ar.shape))
        with open(self.log_path, self.file_mode) as f:
            w = csv.writer(f)
            for row in self.buffer:
                w.writerow(row)

        item_id, values = list(zip(*self.counter.items()))

        item_id, values = np.array(item_id), np.array(values)
        ind_sort = np.argsort(item_id)
        item_id = item_id[ind_sort]
        values = values[ind_sort]

        plt.hist(item_id, weights=values, bins=100, log=True)
        plt.xlabel("Item ID")
        plt.ylabel("Predictions")
        plt.minorticks_off()
        plt.savefig(self.log_path.parent / "prediction_distribution.png")

        with open(self.log_path.parent / "test_prediction_counter.pkl", "wb") as f:
            pickle.dump(self.counter, f)
        plt.clf()

    @property
    def preds(self):
        return np.array(self.buffer)


class MetricsSaver:
    def __init__(self, log_path, file_mode):
        self.log_path = log_path
        self.file_mode = file_mode

    def save_metrics(self, epoch, *args):
        row = [epoch, *args]
        with open(self.log_path, self.file_mode) as f:
            writer = csv.writer(f)
            writer.writerow(row)


class TrajectorySaver:
    def __init__(self, log_path):
        self.log_path = log_path
        self.trajectories = []
        self.current_trajectory = []
        self.time_step = 0
        self.user_id = 0

    def add_step(self, state, action, selected_action, reward):
        selected_action_index = selected_action[0]
        action = action[0].tolist()
        if isinstance(state, ReturnStateTuple):
            state = state.items.flatten().detach().cpu().tolist()
        else:
            state = flatten(state)
        if selected_action_index >= 0:
            selected_action = action[selected_action_index]
        else:
            selected_action = state[-1]
        self.current_trajectory.append((self.user_id, self.time_step, state, action, selected_action, reward[0]))
        self.time_step += 1

    def save_trajectory(self):
        self.trajectories.extend(self.current_trajectory)
        self.current_trajectory = []
        self.time_step = 0
        self.user_id += 1

    def save_to_disk(self):
        df = pd.DataFrame(self.trajectories, columns=["User", "Step", "State", "Action", "Chosen Item", "Reward"])
        print("saving trajectories to {}".format(self.log_path))
        df.to_csv(self.log_path, index=False)

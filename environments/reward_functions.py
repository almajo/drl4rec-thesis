import numpy as np


def target_in_action_reward(predicted_action: np.ndarray, target: int) -> (float, int):
    if target not in predicted_action:
        return 0, -1
    index = np.flatnonzero(predicted_action == target)[0]
    return 1, index

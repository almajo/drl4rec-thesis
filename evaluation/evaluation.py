import logging
from abc import ABC, abstractmethod
from time import time

from tqdm import tqdm

from evaluation.save_callbacks import PredictionSaver, MetricsSaver, TrajectorySaver
from util.decorators import timer

logger = logging.getLogger(__name__)


class Evaluation(ABC):
    name = "AbstractStatic"

    def __init__(self, k):
        self.k = k
        self.model_saver = None

    @abstractmethod
    def get_eval_action(self, obs, k):
        ...

    def get_obs(self, obs):
        return [x[0] for x in obs]

    @timer
    def evaluate(self, env, log_path=None, tb_writer=None, global_step=0, eval_fn=None, scope="test", file_prefix="",
                 file_mode="a+"):
        trajectory_callback = None
        if log_path is not None:
            prefix = file_prefix + "_" if file_prefix else ""
            saver = PredictionSaver(log_path / "{}{}_predictions.csv".format(prefix, scope), file_mode)
            metrics_saver = MetricsSaver(log_path / "{}{}_metrics.csv".format(prefix, scope), file_mode)
            if env.num_envs == 1:
                trajectory_callback = TrajectorySaver(log_path=log_path / "{}{}_trajectories.csv".format(prefix, scope))
            else:
                logger.warning("Not logging trajectories because using more than 1 parallel env")
        if not eval_fn:
            eval_fn = self.get_eval_action
        name = self.agent_name if hasattr(self, "agent_name") else self.name
        logger.info("Evaluating {}".format(name))
        start = time()

        env.hard_reset()

        with tqdm(total=len(env)) as t:
            while 1:
                state = env.reset()
                if state is None:
                    break
                done = False
                while not done:
                    observations = self.get_obs(state)
                    action = eval_fn(observations, k=self.k)
                    state, reward, done, info = env.step(action)
                    if log_path is not None:
                        saver.add_predictions(action)
                    if trajectory_callback is not None:
                        trajectory_callback.add_step(observations, action, info.get("item"), reward)
                    t.update(sum(info.get("dones")))

                if trajectory_callback is not None:
                    trajectory_callback.save_trajectory()

        needed_time = time() - start
        mean_time_per_user = round(needed_time / len(env), 4)
        logger.info("Evaluation Time taken: {}".format(needed_time))
        metrics = env.get_metrics()

        metrics = tuple(map(lambda x: round(x, 3), metrics))

        if log_path is not None:
            metrics_saver.save_metrics(global_step, *metrics, mean_time_per_user)
            saver.save()
            self.saver = saver
            if trajectory_callback is not None:
                trajectory_callback.save_to_disk()
        logger.info("Avg Return: {}, Avg HitRate: {}, List-diversity: {}, Item-diversity: {}".format(*metrics))
        return metrics

    @property
    def predictions(self):
        return self.saver.preds

    def __str__(self):
        return self.name

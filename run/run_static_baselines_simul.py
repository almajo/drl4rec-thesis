import csv
import os
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path
from pprint import pprint

import pandas as pd
import yaml

from environments.parallel_env_wrapper import ParallelEnvWrapper
from environments.recommendation_env import RecommendEnv
from environments.simulator import ParallelSimulator
from models.baselines.static_baselines import *
from run import simulator_values, bpr_iteration_version
from util.helpers import set_random_seeds

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s -  %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

result = namedtuple("results", "model dataset avg_return hitrate list_diversity item_diversity")

DATA_PATH = Path(os.getenv("DATA_PATH", "/home/stud/grimmalex/datasets"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/home/stud/grimmalex/thesis/output"))
parser = ArgumentParser()
parser.add_argument("data", choices=["ml/ml-1m", "ml/ml-25m", "taobao"], nargs="+")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--experiment_name", default="test")
parser.add_argument("--replicas", type=int, default=128)
parser.add_argument("--reward", choices=["item", "list"], default="item")
args = parser.parse_args()

# Seed everything, including torch
set_random_seeds(args.seed)
reward = args.reward
num_replicas = args.replicas

baselines = [ItemKNN, RandomBaseline, PopularBaselineAdaptive]
datasets = args.data
result_list = []
metrics = []
debug = False
simulator_type = "bpr"
simulator_values["reward_type"] = reward


def evaluate_one_agent(agent: Evaluation, test_simulator, sim_type, data_name, num_seeds=3, log_path=None):
    inner_metrics = []
    for i in range(num_seeds):
        np.random.seed(i)
        m = agent.evaluate(test_simulator, file_prefix=sim_type, file_mode="a+", log_path=log_path)
        rewards = test_simulator.envs[0].all_mean_rewards
        print(np.mean(rewards))
        inner_metrics.append(m)
    inner_metrics = np.array(inner_metrics)
    mean, std = inner_metrics.mean(axis=0), inner_metrics.std(axis=0)
    pairs = ["{:.3f} ({:.3f})".format(mu, st) for mu, st in zip(mean, std)]
    pairs = [agent.name, data_name] + pairs

    full_values = [[agent.name, data_name, seed] + met for seed, met in enumerate(inner_metrics.tolist())]

    return pairs, full_values


raw_metrics = []
base_path = OUTPUT_PATH / "baselines"
for d in datasets:
    log_dir = base_path / d
    log_dir.mkdir(parents=True, exist_ok=True)
    print(log_dir)
    data_dir = DATA_PATH / d
    top_k_actions = 10
    max_episode_len = 0
    max_state_len = 10
    num_parallel_envs = 1
    test_max_len = 100

    num_eval_seeds = 5

    train_env = RecommendEnv().initialize(data_dir / "train_split.csv",
                                          num_repeats=1,
                                          sample_k=top_k_actions,
                                          max_episode_len=max_episode_len,
                                          max_state_len=max_state_len,
                                          )
    env = ParallelEnvWrapper(train_env, num_parallel_envs)
    valid_env = ParallelSimulator(data_dir / "valid_split.csv",
                                  num_replicas=num_parallel_envs,
                                  max_state_len=max_state_len,
                                  max_traj_len=test_max_len, variant="logs")

    # For log-data only
    # For simulator

    test_env = ParallelSimulator(data_dir / "test.csv", num_replicas=num_parallel_envs, max_state_len=max_state_len,
                                 max_traj_len=test_max_len, variant=simulator_type,
                                 iteration_level=bpr_iteration_version.get(log_dir.name),
                                 **simulator_values)

    metric_list = []
    for ba in baselines:
        logging = None
        if "knn" in ba.__name__.lower():
            logging = log_dir / ba.__name__
        b = ba(env, k=top_k_actions, tf_idf=True, log_dir=logging)
        new_metrics, raw = evaluate_one_agent(b, test_env, simulator_type, data_dir.name, num_seeds=num_eval_seeds,
                                              log_path=logging)
        metric_list.append(new_metrics)
        raw_metrics.extend(raw)

    # save metrics
    save_path = log_dir / "{}_mean_performance_test.csv".format(simulator_type)
    print("Saving metrics to {}".format(save_path))
    with open(save_path, "w+") as f:
        writer = csv.writer(f)
        writer.writerows(metric_list)

settings = {
    "simulator_values": simulator_values,
    "bpr_version": bpr_iteration_version
}
with open(base_path / "simulation_settings.yaml", "w+") as f:
    yaml.dump(settings, f)

df = pd.DataFrame(raw_metrics, columns=["Method", "Dataset", "Seed", "Return", "Hitrate", "List-Diversity", "Coverage"])
df.to_csv(base_path / "{}_{}_test_new.csv".format(reward, simulator_type), index=False)
pprint(df)
# save_to_csv(metrics, base_path / "results-baselines-{}.csv".format(simulator_type))

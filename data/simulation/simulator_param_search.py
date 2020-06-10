import os
from argparse import ArgumentParser
from collections import namedtuple
from itertools import product
from pathlib import Path
from pprint import pprint

import pandas as pd

from environments.parallel_env_wrapper import ParallelEnvWrapper
from environments.recommendation_env import RecommendEnv
from environments.simulator import ParallelSimulator
from models.baselines.static_baselines import *
from run import bpr_iteration_version
from util.helpers import set_random_seeds

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s -  %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

result = namedtuple("results", "model dataset avg_return hitrate list_diversity item_diversity")

DATA_PATH = Path(os.getenv("DATA_PATH", "/home/stud/grimmalex/datasets"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/home/stud/grimmalex/thesis/output"))
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--experiment_name", default="test")
parser.add_argument("--replicas", type=int, default=128)
parser.add_argument("--reward", choices=["item", "list"], default="item")
args = parser.parse_args()

# Seed everything, including torch
set_random_seeds(args.seed)
reward = args.reward
num_replicas = args.replicas

baselines = [RandomBaseline, PopularBaselineAdaptive]
# baselines = [ItemKNN, RandomBaseline, PopularBaselineAdaptive]
result_list = []
debug = False
simulator_type = "bpr"

raw_metrics = []
base_path = OUTPUT_PATH / "baselines"

# param-search values
default_leave = [0.05, 0.2]
other_item_reward = [1.5]
other_item_leave = [0.2, 0.3, 0.4, 0.5]
repetition_factor = [0.1, 0.2, 0.3]

d = "ml/ml-1m"

hyper_param_list = []

for leave_prob, other_leave, other_reward, rep in product(default_leave, other_item_leave, other_item_reward,
                                                          repetition_factor):
    log_dir = base_path / d
    log_dir.mkdir(parents=True, exist_ok=True)
    print(log_dir)
    data_dir = DATA_PATH / d
    top_k_actions = 10
    max_episode_len = 0
    max_state_len = 10
    num_parallel_envs = 1
    test_max_len = 100

    num_eval_seeds = 1

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
    simulator_values = {
        "default_leave_prob": leave_prob,
        "other_item_leave_prob": other_leave,
        "other_item_reward": other_reward,
        "repetition_factor": rep,
        "reward_type": "item"
    }

    test_env = ParallelSimulator(data_dir / "test.csv", num_replicas=num_parallel_envs, max_state_len=max_state_len,
                                 max_traj_len=test_max_len, variant=simulator_type,
                                 iteration_level=bpr_iteration_version.get(log_dir.name),
                                 **simulator_values)

    for ba in baselines:
        b = ba(env, k=top_k_actions, tf_idf=True, log_dir=Path("/tmp"))
        return_, *_ = b.evaluate(test_env, file_prefix=simulator_type, file_mode="a+")
        hyper_param_list.append((b.name, return_, leave_prob, other_leave, other_reward, rep))

df = pd.DataFrame(hyper_param_list,
                  columns=["Method", "return", "leave_click", "leave_no_click", "other_reward", "zeta"])
pprint(df)
df.to_csv(base_path / d / "simulator_param_search.csv")

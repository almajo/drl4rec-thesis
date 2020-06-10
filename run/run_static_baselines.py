import os
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from environments.parallel_env_wrapper import ParallelEnvWrapper
from environments.recommendation_env import RecommendEnv
from environments.simulator import ParallelSimulator
from models.baselines.static_baselines import *
from util.helpers import set_random_seeds
from util.in_out import save_to_csv

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s -  %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

result = namedtuple("results", "model dataset avg_return hitrate list_diversity item_diversity")

DATA_PATH = Path(os.getenv("DATA_PATH", "/home/stud/grimmalex/datasets"))

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--experiment_name", default="test")
parser.add_argument("--replicas", type=int, default=128)
parser.add_argument("--embedding", default="standard", choices=["standard", "genome", "word2vec"])
parser.add_argument("data", nargs="+", choices=["ml/ml-1m", "ml/ml-25m", "taobao"])
parser.add_argument("--path")

args = parser.parse_args()

# Seed everything, including torch
set_random_seeds(args.seed)

num_replicas = args.replicas

baselines = [ItemKNN, PopularBaseline, PopularBaselineAdaptive]
datasets = args.data
result_list = []
metrics = []
debug = False

if not args.path:
    log_dir = Path("/home/stud/grimmalex/thesis/output/baselines")
else:
    log_dir = Path(args.path)

log_dir.mkdir(parents=True, exist_ok=True)
for d in datasets:
    print(d)
    data_dir = DATA_PATH / d
    top_k_actions = 10
    max_episode_len = 0
    max_state_len = 10
    num_parallel_envs = 1
    test_max_len = 100

    train_env = RecommendEnv().initialize(data_dir / "train_split.csv",
                                          num_repeats=1,
                                          sample_k=top_k_actions,
                                          max_episode_len=max_episode_len,
                                          max_state_len=max_state_len,
                                          )
    env = ParallelEnvWrapper(train_env, num_parallel_envs)
    valid_env = ParallelSimulator(data_dir / "valid_split.csv", num_replicas=num_parallel_envs,
                                  max_state_len=max_state_len,
                                  max_traj_len=test_max_len, variant="logs")

    # For log-data only
    # For simulator
    test_env = ParallelSimulator(data_dir / "test.csv", num_replicas=num_parallel_envs, max_state_len=max_state_len,
                                 max_traj_len=test_max_len, variant="logs")

    for ba in baselines:
        logging = log_dir / d / ba.__name__
        logging.mkdir(exist_ok=True, parents=True)
        b = ba(env, top_k_actions, tf_idf=True, log_dir=logging)
        new_metrics = b.evaluate(test_env, log_path=logging)
        print(new_metrics)
        metrics.append(result(ba.__name__, data_dir.name, *new_metrics))

save_to_csv(metrics, log_dir / "results-baselines-{}.csv".format(datetime.now().time()))

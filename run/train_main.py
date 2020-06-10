import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
import yaml

from drlap.utilities.data_structures.Config import Config
from environments.parallel_env_wrapper import ParallelEnvWrapper
from environments.recommendation_env import RecommendEnv
from environments.simulator import ParallelSimulator
from models import AGENTS
from util.Trainer import Trainer
from util.helpers import set_random_seeds

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s -  %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path(os.getenv("DATA_PATH", "/home/stud/grimmalex/datasets"))
BASE_PATH = Path(os.getenv("BASE_PATH", "/home/stud/grimmalex/thesis/output"))
CONFIG_PATH = Path(os.path.dirname(os.path.realpath(__file__))) / "config_files"

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--experiment_name", default="test")
parser.add_argument("--agent", default="wolpertinger")
parser.add_argument("--data", default="ml/ml-1m")
parser.add_argument("--num_iterations", default=2000, type=int)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--subdir", default="")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--word2vec", type=int, default=0)
parser.add_argument("--continue_training", action="store_true", default=False)
parser.add_argument("--pretrain", default="")
parser.add_argument("--batch", action="store_true")
parser.add_argument("--num_episodes", default=10_000, type=int)
parser.add_argument("--simulator", action="store_true")
parser.add_argument("--gamma", type=float, default=0.6)

args = parser.parse_args()
np.seterr(all='raise')
# ensure the cli is used correctly
if args.pretrain == "full":
    assert args.word2vec
if args.batch:
    assert (args.word2vec or "25m" in args.data) and not args.pretrain

# Seed all possible random number generators for reproducibility
set_random_seeds(args.seed)

logger.info("\n" + pformat(args.__dict__))

num_iterations = args.num_iterations

agent = AGENTS.get(args.agent.lower())
if not agent:
    raise ModuleNotFoundError("Did not find the agent {}, choose one from \n{}".format(args.agent, list(AGENTS.keys())))

if args.subdir:
    BASE_PATH /= args.subdir

data_name = args.data
data_file_name = data_name.split("/")[-1]
experiment_save_path = BASE_PATH / "{}/{}/{}/{}".format(args.experiment_name, data_file_name, args.agent,
                                                        args.seed)
experiment_save_path.mkdir(parents=True, exist_ok=True)
data_dir = DATA_PATH / args.data

#  Variables for env and training length
top_k_actions = 10
max_episode_len = 0
max_state_len = 10
num_parallel_envs = 32
max_episode_len_test = 100
num_episodes = args.num_episodes

reward_type = "item"
###

if not args.simulator:
    train_env = RecommendEnv(debug=args.debug).initialize(data_dir / "train_split.csv",
                                                          num_repeats=num_iterations,
                                                          sample_k=top_k_actions,
                                                          max_episode_len=max_episode_len,
                                                          max_state_len=max_state_len,
                                                          )
    env = ParallelEnvWrapper(train_env, num_parallel_envs)
    sim_type = "logs"
else:
    env = ParallelSimulator(data_dir / "train_split.csv",
                            num_replicas=num_parallel_envs,
                            max_state_len=max_state_len,
                            max_traj_len=max_episode_len_test,
                            variant="bpr",
                            simulation_type="train",
                            reward_type=reward_type)
    sim_type = "bpr"

valid_env = ParallelSimulator(data_dir / "valid_split.csv",
                              num_replicas=num_parallel_envs,
                              max_state_len=max_state_len,
                              max_traj_len=max_episode_len_test,
                              variant=sim_type,
                              simulation_type="valid",
                              iteration_level=0)

test_env = ParallelSimulator(data_dir / "test.csv",
                             num_replicas=num_parallel_envs,
                             max_state_len=max_state_len,
                             max_traj_len=max_episode_len_test,
                             variant=sim_type,
                             reward_type=reward_type,
                             simulation_type="test",
                             iteration_level=9)

# valid_env = test_env

config = Config()
config.seed = args.seed
config.environment = env
config.valid_environment = valid_env
config.test_environment = test_env
config.num_episodes_to_run = num_episodes
config.model_save_interval = min(len(env) // num_parallel_envs, 200)  # save model every n epochs, overwriting oneself
config.evaluation_interval = min(len(env) // num_parallel_envs, 500)
config.file_to_save_model = str(experiment_save_path / "model.pkl")
config.tb_dir = str(experiment_save_path)
config.tb_log_interval = 20
config.show_solution_score = False
config.use_GPU = torch.cuda.is_available() and not args.cpu
logger.info("Using GPU: {}".format(config.use_GPU))
config.overwrite_existing_results_file = True
config.randomise_random_seed = False
config.save_model = True
config.render_from = -1
config.metrics_k = top_k_actions
config.pretrained_model_base_dir /= data_file_name

# embedding info to schedule independently
w2v_embedding = args.word2vec
embedding_dim = 16
if w2v_embedding:
    embedding = "word2vec_fixed"
elif "ml-25m" in data_file_name:
    embedding = "genome"
    embedding_dim = 32
else:
    embedding = "standard"

w2v_context_path = str(data_dir / "word2vec/tensor_{}.pkl".format(w2v_embedding or 32))
tag_genome_embedding_path = str(data_dir / "pca_genome_embedding_32.pkl")

pretrain_state = args.pretrain == "state"
pretrain_full = args.pretrain == "full"
default_params = {
    "Embedding": {
        "type": embedding,
        "embedding_dim": embedding_dim,
        "freeze": embedding != "standard",
        "num_items": env.action_space.n,
        "sparse_grad": False,  # Faster but does not allow weight decay (would need to normalize over all rows)
        "scale_by_freq": False,  # not supported with sparse grads
        "w2v_context_path": w2v_context_path,
        "tag_genome_embedding_path": tag_genome_embedding_path
    },
    "State": {
        "type": "rnn",
        "rnn_dim": 128,
        "save_path": str(experiment_save_path / "pretrained_state.pkl")
    },
    "batch_rl": args.batch,
    "simulator": args.simulator,
    "load_state_module": False,
    "state-only-pretrain": pretrain_state,
    "pretrain": pretrain_full,
    "pretrain_steps": 3000,
    "pretrain_eval_steps": 250,
    "history_masking": False,
    "batch_size": 64,
    "test_batch_size": 64,
    "min_steps_before_learning": 0,
    "epsilon_sample_size": 500,
    "static_epsilon": 0.25 if not args.batch else 0.,
    "Epsilon": {
        "decay": False,  # if set to False, will use the state_epsilon above
        "start_value": 1,
        "end_value": 0.1,
        "end_episode": int(num_episodes * 0.5)
    },
    "continue_training": args.continue_training,
    "buffer_type": "priority"  # use default as an alternative [default|priority]
}

name = agent.agent_name.lower()
# name = "dqn" if "dqn" in agent_name else agent_name
with open(CONFIG_PATH / (name + ".json")) as f:
    agent_parameters = json.load(f)
type_key = list(agent_parameters.keys())[0]
agent_dict = agent_parameters[type_key]
for k, v in default_params.items():
    # the config file parameters can not be overwritten
    if k not in agent_dict:
        agent_dict[k] = v
    else:
        logger.warning(
            "Parameter {} is specified in config file. file-value:{}, default-value_{}".format(k, agent_dict[k],
                                                                                               default_params[k]))

config.hyperparameters = agent_parameters

if "DQN_Agents" in config.hyperparameters:
    config.hyperparameters["DQN_Agents"]["discount_rate"] = args.gamma
elif "Actor_Critic_Agents" in config.hyperparameters:
    config.hyperparameters["Actor_Critic_Agents"]["discount_rate"] = args.gamma
elif "Policy_Gradient_Agents" in config.hyperparameters:
    config.hyperparameters["Policy_Gradient_Agents"]["discount_rate"] = args.gamma

with open(experiment_save_path / "hyperparameters.yaml", "w+") as f:
    yaml.dump(config.hyperparameters, f, default_flow_style=False)


def start_training():
    trainer = Trainer(config, agent)
    trainer.run_games_for_agent()


if __name__ == "__main__":
    start_training()

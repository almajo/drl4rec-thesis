import os
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import yaml

from drlap.utilities.data_structures.Config import Config
from environments.simulator import ParallelSimulator
from models import AGENTS
from run import *
from util.helpers import set_random_seeds

global_configs = dict()


def load_environments(trajectory_file, simulation_dir, simulator_variant="bpr", train_file=None,
                      valid_env=False):
    max_state_len = 10
    num_parallel_envs = 1
    test_max_len = 100

    test_env = ParallelSimulator(trajectory_file, simulation_type=simulation_dir, num_replicas=num_parallel_envs,
                                 max_state_len=max_state_len,
                                 max_traj_len=test_max_len, variant=simulator_variant,
                                 iteration_level=bpr_iteration_version.get(trajectory_file.parent.name),
                                 **simulator_values)
    train_env = ParallelSimulator(train_file or trajectory_file, simulation_type=simulation_dir,
                                  num_replicas=num_parallel_envs,
                                  max_state_len=max_state_len,
                                  max_traj_len=test_max_len, variant="logs")
    valid_env = ParallelSimulator(trajectory_file.parent / "valid_split.csv", simulation_type=simulation_dir,
                                  num_replicas=num_parallel_envs,
                                  max_state_len=max_state_len,
                                  max_traj_len=test_max_len, variant="logs") if valid_env else None
    return train_env, test_env, valid_env


def get_base_config(train_env, test_env, valid_env, experiment_dir, seed):
    top_k_actions = 10

    config = Config()
    config.seed = seed
    config.environment = train_env
    config.test_environment = test_env
    config.valid_environment = valid_env
    config.num_episodes_to_run = 1
    config.model_save_interval = 1000000000  # save model every n epochs, overwriting oneself
    config.evaluation_interval = 5000000000
    config.file_to_save_model = str(experiment_dir / "model.pkl")
    config.tb_dir = str(experiment_dir)
    config.tb_log_interval = 10
    config.show_solution_score = False
    config.use_GPU = torch.cuda.is_available()
    config.overwrite_existing_results_file = True
    config.randomise_random_seed = False
    config.save_model = False
    config.render_from = -1
    config.metrics_k = top_k_actions
    config.use_tb = False
    return config


def load_model(model_dir, sim="test", simulator="bpr"):
    global global_configs
    data, agent, seed = str(model_dir).split("/")[-3:]
    seed = int(seed)
    if "ml" in data:
        data = "ml/{}".format(data)
    data_dir = DATA_PATH / data

    trajectory_file = data_dir / "test.csv"

    agent = AGENTS.get(agent)
    if not agent:
        raise ModuleNotFoundError(
            "Did not find the agent {}, choose one from \n{}".format(args.agent, str(AGENTS.keys())))
    if data not in global_configs:
        envs = load_environments(trajectory_file, sim, simulator_variant=simulator)
        global_configs[data] = envs
    else:
        envs = global_configs[data]
    config = get_base_config(*envs, model_dir, seed)

    with open(model_dir / "hyperparameters.yaml", "r") as f:
        hyperparameters = yaml.load(f, yaml.Loader)
    main_key = list(hyperparameters.keys())[0]
    config.hyperparameters = hyperparameters[main_key]

    w2v_path = config.hyperparameters["Embedding"]["w2v_context_path"]
    if str(DATA_PATH) not in w2v_path:
        end_path = str(w2v_path).split("datasets/")[-1]
        w2v_path = DATA_PATH / end_path
        config.hyperparameters["Embedding"]["w2v_context_path"] = w2v_path
    agent = agent(config)
    return agent


def evaluate_one_agent(agent, num_seeds=3):
    metrics = []
    for i in range(num_seeds):
        set_random_seeds(i)
        m = agent.evaluate(file_prefix=sim_type, file_mode="a+", return_all_metrics=True)
        metrics.append(m)
    metrics = np.array(metrics)
    mean, std = metrics.mean(axis=0), metrics.std(axis=0)
    pairs = ["{:.3f} ({:0.3f})".format(mu, st) for mu, st in zip(mean, std)]
    pairs = [agent.agent_name, agent.config.environment.id] + pairs

    full_values = [[agent.agent_name, agent.config.environment.id, seed] + met for seed, met in
                   enumerate(metrics.tolist())]
    return pairs, full_values


def get_best_seed_path(model_base_dir):
    max_val = -1
    max_path = None
    for d in sorted(model_base_dir.glob("**/best_checkpoint.txt")):
        print(d)
        with open(d) as file:
            content = file.read().strip()
        metric_val = content.split("_")[-1][:-5]
        metric_val = float(metric_val)
        if metric_val > max_val:
            max_path = d.parent / content
    return max_path


def eval_agents_from_checkpoint(experiment_dir, num_seeds):
    agent_metrics = []
    name = {"ml-1mlird", "ml-25mlird", "taobaolird"}
    all_agent_metrics = []
    for d in sorted(experiment_dir.glob("**/valid_metrics.csv")):
        method = d.parent.parent.name
        data = d.parent.parent.parent.name
        mn = data + method
        if mn in name:
            continue
        agent = load_model(d.parent, simulator=sim_type)

        agent.set_mask(True)

        max_seed_path = get_best_seed_path(d.parent.parent)
        print(max_seed_path)
        agent.load_pretrained_models(max_seed_path)
        ag_metrics, raw = evaluate_one_agent(agent, num_seeds)
        agent_metrics.append(ag_metrics)
        all_agent_metrics.extend(raw)
        name.add(mn)
    return agent_metrics, all_agent_metrics


def eval_prediction_saving(experiment_dir):
    for d in experiment_dir.glob("**/hyperparameters.yaml"):
        agent = load_model(d.parent, simulator=sim_type)
        agent.load_pretrained_models()
        # agent.evaluate(file_mode="a+")
        agent.evaluate(file_prefix=sim_type, file_mode="a+")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("type", choices=["logs", "bpr", "gru"])
    parser.add_argument("--reward", choices=["item", "list"], default="item")
    args = parser.parse_args()
    reward_type = args.reward
    sim_type = args.type
    experiment_path = Path(args.model_dir)
    DATA_PATH = Path(os.getenv("DATA_PATH", "/home/stud/grimmalex/datasets"))

    simulator_values["reward_type"] = reward_type

    # eval_prediction_saving(experiment_path)
    seeds = 1

    all_metrics, raw_metrics = eval_agents_from_checkpoint(experiment_path, seeds)

    # save metrics
    save_path = experiment_path / "simulator_param_test.csv"
    print("Saving metrics to {}".format(save_path))

    df = pd.DataFrame(all_metrics, columns=["Method", "Dataset", "Return", "Hitrate", "List-Diversity", "Coverage"])
    pprint(df.Return)
    # df.to_csv(save_path, index=False)
    #
    # df = pd.DataFrame(raw_metrics,
    #                   columns=["Method", "Dataset", "Seed", "Return", "Hitrate", "List-Diversity", "Coverage"])
    # df.to_csv(save_path.parent / "{}_{}_test_new.csv".format(reward_type, sim_type), index=False)
    # pprint(df)

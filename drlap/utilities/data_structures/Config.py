from copy import copy
from pathlib import Path

import gym
import yaml


class Config(object):
    """Object to hold the config requirements for an agent/game"""

    def __init__(self):
        self.seed = None
        self.environment = None
        self.test_environment = None
        self.valid_environment = None
        self.requirements_to_solve_game = None
        self.num_episodes_to_run = None
        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.file_to_save_model = None
        self.runs_per_agent = None
        self.visualise_overall_results = None
        self.visualise_individual_results = None
        self.hyperparameters = None
        self.use_GPU = None
        self.overwrite_existing_results_file = None
        self.save_model = False
        self.standard_deviation_results = 1.0
        self.randomise_random_seed = True
        self.show_solution_score = False
        self.debug_mode = False
        self.tb_dir = False
        self.tb_log_interval = 1
        self.model_save_interval = 100
        self.render_from = -1
        self.evaluation_interval = -1
        self.metrics_k = 20
        self.pretrained_model_base_dir = Path("/home/stud/grimmalex/thesis/pretrained_models")
        self.use_tb = True

    def save(self, save_dir):
        file_name = str(Path(save_dir) / "config.yaml")
        env_name = self.file_to_save_data_results.split("/")[-3]
        config = copy(self)
        config.environment = env_name
        config.test_environment = None

        with open(file_name, 'w') as yaml_file:
            yaml.dump(config.__dict__, yaml_file, default_flow_style=False)

    def eval_mode(self):
        try:
            self.environment = gym.make(self.environment)
        except gym.error.UnregisteredEnv:
            import roboschool
            self.environment = gym.make(self.environment)
        self.tb_dir = None
        self.use_GPU = False

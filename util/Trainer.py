import copy
import logging
import random
from pprint import pformat

logger = logging.getLogger(__name__)


def print_two_empty_lines():
    logger.info("-----------------------------------------------------------------------------------")
    logger.info("-----------------------------------------------------------------------------------")
    logger.info(" ")


class Trainer(object):
    """Runs games for given models. Optionally will visualise and save the results"""

    def __init__(self, config, agent):
        self.config = config
        self.agent = agent
        self.agent_to_agent_group = self.create_agent_to_agent_group_dictionary()

    @staticmethod
    def create_agent_to_agent_group_dictionary():
        """Creates a dictionary that maps an agent to their wider agent group"""
        return {
            "DQN": "DQN_Agents",
            "DQN-HER": "DQN_Agents",
            "DDQN": "DQN_Agents",
            "DDQN with Prioritised Replay": "DQN_Agents",
            "DQN with Fixed Q Targets": "DQN_Agents",
            "Duelling DQN": "DQN_Agents",
            "PPO": "Policy_Gradient_Agents",
            "REINFORCE": "Policy_Gradient_Agents",
            "Genetic_Agent": "Stochastic_Policy_Search_Agents",
            "Hill Climbing": "Stochastic_Policy_Search_Agents",
            "DDPG": "Actor_Critic_Agents",
            "DDPG-HER": "Actor_Critic_Agents",
            "TD3": "Actor_Critic_Agents",
            "A2C": "Actor_Critic_Agents",
            "A3C": "Actor_Critic_Agents",
            "h-DQN": "h_DQN",
            "SNN-HRL": "SNN_HRL",
            "HIRO": "HIRO",
            "SAC": "Actor_Critic_Agents",
            "HRL": "HRL",
            "Model_HRL": "HRL",
            "DIAYN": "DIAYN",
            "Dueling DDQN": "DQN_Agents",
            "Wolpertinger": "Actor_Critic_Agents",
            "TPGR": "Policy_Gradient_Agents",
            "PDQN": "DQN_Agents",
            "PDDQN": "DQN_Agents",
            "PDDDQN": "DQN_Agents",
            "Dueling": "DQN_Agents",
            "AQL": "Actor_Critic_Agents",
            "TAQL": "Actor_Critic_Agents",
            "LIRD": "Actor_Critic_Agents",
            "Correction": "Policy_Gradient_Agents",
            "CorrectionPPO": "Policy_Gradient_Agents",
            "GRU": "Policy_Gradient_Agents",
            "NEWS": "DQN_Agents",
            "REM": "DQN_Agents",
            "QR-DQN": "DQN_Agents"

        }

    @staticmethod
    def set_kill_signal_handler(agent):
        import signal
        import sys

        def signal_handler(sig, frame):
            agent.locally_save_policy()
            sys.exit(-1)

        signal.signal(signal.SIGTERM, signal_handler)

    def run_games_for_agent(self):
        """Runs a set of games for a given agent, saving the results in self.results"""
        agent_name = self.agent.agent_name
        agent_group = self.agent_to_agent_group[agent_name]
        agent_config = copy.deepcopy(self.config)
        if self.config.randomise_random_seed:
            agent_config.seed = random.randint(0, 2 ** 32 - 2)
        agent_config.hyperparameters = agent_config.hyperparameters[agent_group]
        logger.info("AGENT NAME: {}".format(agent_name))
        agent = self.agent(agent_config)

        self.set_kill_signal_handler(agent)

        logger.info(pformat(agent.hyperparameters))
        logger.info("RANDOM SEED {}".format(agent_config.seed))
        agent.run_n_episodes()

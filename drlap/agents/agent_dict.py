from drlap.agents.DQN_agents import *
from drlap.agents.actor_critic_agents import *
from models.rl.q_learning import ParameterizedDQNAgent, ParameterizedDDQNAgent
from models.rl.wolpertinger.Wolpertinger import Wolpertinger

actor_critic = [DDPG, DDPG_HER, TD3, AKTR, SAC_Discrete, SAC, Wolpertinger]
q = [DDQN, DDQN_With_Prioritised_Experience_Replay, DQN_HER, Dueling_DDQN, DQN, ParameterizedDQNAgent,
     ParameterizedDDQNAgent]

all_agents = actor_critic + q

Agents = {a.__name__: a for a in all_agents}
Agents["dqn"] = ParameterizedDQNAgent
Agents["ddqn"] = ParameterizedDDQNAgent

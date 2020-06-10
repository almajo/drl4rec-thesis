from models.baselines.gru_rec import GRU
from models.rl import QRDQN
from models.rl import REM
from models.rl.correction import CorrectionReinforce
from models.rl.correction import CorrectionReinforcePPO
from models.rl.lird import LIRD
from models.rl.q_learning import ParameterizedDDQNAgent, ParameterizedDQNAgent, \
    ParameterizedDuellingDDQNAgent
from models.rl.q_learning.news import NewsReco
from models.rl.q_learning.q import DDQNAgent, DQNAgent, Dueling
from models.rl.tpgr import TPGR
from models.rl.wolpertinger.Wolpertinger import Wolpertinger

AGENTS = {
    "tpgr": TPGR,
    "wolpertinger": Wolpertinger,
    "pddqn": ParameterizedDDQNAgent,
    "pdddqn": ParameterizedDuellingDDQNAgent,
    "pdqn": ParameterizedDQNAgent,
    "duelling": Dueling,
    "dueling": Dueling,
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "lird": LIRD,
    "correction": CorrectionReinforce,
    "correctionppo": CorrectionReinforcePPO,
    "news": NewsReco,
    "gru": GRU,
    "rem": REM,
    "qrdqn": QRDQN
}

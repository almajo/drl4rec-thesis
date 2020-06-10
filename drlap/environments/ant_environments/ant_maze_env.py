# NOTE THIS CODE IS TAKEN FROM https://github.com/tensorflow/models/tree/master/research/efficient-hrl/environments
# and is not my code.


from ant_environments.ant import AntEnv
from ant_environments.maze_env import MazeEnv


class AntMazeEnv(MazeEnv):
    MODEL_CLASS = AntEnv

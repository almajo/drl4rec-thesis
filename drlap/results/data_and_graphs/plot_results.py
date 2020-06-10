from drlap.agents.Trainer import Trainer
from drlap.utilities.data_structures.Config import Config

trainer = Trainer(config=Config(), agents=None)

trainer.visualise_preexisting_results(save_image_path="test-1/graph_results.png", data_path="test-1/data_results.pkl",
                                      title="MountainCar")

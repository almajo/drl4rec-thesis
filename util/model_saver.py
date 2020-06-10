from pathlib import Path

import torch


class ModelSaver:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.model_dir = base_dir / "models"
        self.overview_file = self.model_dir / "last_checkpoint.txt"
        self.eval_best_file = self.model_dir / "best_checkpoint.txt"
        self.best_valid_metric = float("-inf")
        if self.eval_best_file.exists():
            self._load_best_value()

    def save_model(self, dictionary, episode_number, metric_value, scope="train"):
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True)
        file_name = self.model_dir / "model_{}_{:.3f}.pkl".format(episode_number, metric_value)
        torch.save(dictionary, file_name)
        print("Saved model to {}".format(file_name))
        if scope != "valid" and scope != "test":
            # save the path to the last checkpoint for continue training
            with open(str(self.overview_file), "w+") as f:
                f.write(file_name.name)
        elif scope == "valid":
            # save the path to the model which performed best on valid during training
            if metric_value >= self.best_valid_metric:
                with open(str(self.eval_best_file), "w+") as f:
                    f.write(file_name.name)
                self.best_valid_metric = metric_value

    def _load_best_value(self):
        p = str(self.get_best_model_path())
        value = p.split("_")[-1][:-4]
        self.best_valid_metric = float(value)

    def model_exists(self):
        return self.overview_file.exists() or self.eval_best_file.exists()

    def get_best_model_path(self):
        with open(str(self.eval_best_file), "r") as f:
            path = f.read().strip()
        if Path(path).is_absolute():
            path = Path(path).name
        return self.eval_best_file.parent / path

    def get_last_checkpoint_path(self):
        p = self.overview_file if self.overview_file.exists() else self.eval_best_file
        with open(str(p), "r") as f:
            path = f.read().strip()
        if Path(path).is_absolute():
            # compatibility for full path old logs
            path = Path(path).name
        return p.parent / path

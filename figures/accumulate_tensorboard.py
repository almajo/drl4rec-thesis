import logging
import warnings
from argparse import ArgumentParser
from multiprocessing.pool import Pool
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.WARNING)
warnings.simplefilter(action='ignore', category=FutureWarning)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_one_dir(path: Path):
    x = EventAccumulator(path=str(path))
    x.Reload()
    x.FirstEventTimestamp()
    keys = {
        'train/running_return': "Return",
        'debug/expected_q_values': "Q-values"
    }
    df = None
    for k, v in keys.items():
        try:
            time_steps = x.Scalars(k)
        except KeyError:
            logging.warning("Did not find the key in {}".format(path))
            continue
        wall_time, steps, values = list(zip(*time_steps))
        df_new = pd.DataFrame(data={
            "Epoch": steps,
            v: values
        })
        if df is None:
            df = df_new
        else:
            df = df.merge(df_new, on="Epoch")

    experiment_name, data, method, seed = str(path).split("/")[-4:]
    n = len(df)

    df["Method"] = [method] * n
    df["Experiment"] = [experiment_name] * n
    df["Seed"] = [seed] * n
    df["Dataset"] = [data] * n
    return df


def accumulate_directory(base_dir):
    directories = [p.parent for p in base_dir.glob("**/events*")]
    with Pool() as p:
        output = p.map(load_one_dir, directories)
    df = pd.concat(output)
    df.to_csv(base_dir / "training_tb_metrics.csv", index=False)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("base_dir")
    args = p.parse_args()
    base = Path(args.base_dir)
    accumulate_directory(base)

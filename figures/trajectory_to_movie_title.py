import os
import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser()
parser.add_argument("base_path")
args = parser.parse_args()

base_path = Path(args.base_path)
DATA_PATH = Path(os.getenv("DATA_PATH", "/home/stud/grimmalex/datasets"))

tags = pd.read_csv(DATA_PATH / "ml/ml-25m/content_prep.csv", usecols=[0, 1], index_col=[0], squeeze=True)
tags = tags.map(lambda x: re.sub("\(\d{4}\)", "", x).strip())
for p in base_path.glob("**/*trajectories.csv"):
    print(p)
    df = pd.read_csv(p, converters={"State": eval, "Action": eval})
    df.loc[:, "Action"] = df.Action.map(lambda x: [tags.get(idx, "NA") for idx in x])
    df.loc[:, "State"] = df.State.map(lambda x: [tags.get(idx, "NA") for idx in x])
    df.to_csv(p.parent / "simulation_traj_titles.csv", index=False)

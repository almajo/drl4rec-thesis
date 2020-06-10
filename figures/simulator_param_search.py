import pandas as pd

DATA_PATH = "/home/alex/workspace/uni_output/baselines/ml/ml-1m/simulator_param_search.csv"
df = pd.read_csv(DATA_PATH)

df = df.loc[(df.leave_click == 0.05) & (df.other_reward == 1.5) & (df.zeta == 0.2)]
print(df)

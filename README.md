### Model-Free Deep Reinforcement Learning for Interactive Recommender Systems: An Evaluation 

This project contains the research code used in my Master's Thesis.

It is a critic evaluation of different published algorithms on ONE environment, using publicly available datasets. The recommender domain in general suffers from the problem of missing comparability and reproducibility. Hence, I chose it as my master thesis' goal to compare them in an offline and online evaluation setting.

#### Involved algorithms
 - [TPGR (Tree-based policy gradient)](https://arxiv.org/abs/1811.05869)
 - [Top-K Off-Policy Correction](https://dl.acm.org/doi/abs/10.1145/3289600.3290999?casa_token=nAl4__wjMn8AAAAA:AM6VEJxARhjLWjXyxxNWSvDFJ1yufd7sKU_aBSCJl6KM2_PgxlqFY39gt3-xFxGIlbeIcVo8WEKyh2o)
 - [Wolpertinger (alias KNN-Actor-Critic)](https://arxiv.org/abs/1512.07679)
 - [LIRD (List-Wise Recommendations)](https://arxiv.org/abs/1801.00209)
 - [NEWS (Parametric DQN)](https://dl.acm.org/doi/abs/10.1145/3178876.3185994?casa_token=Fg-VXbqUkuUAAAAA:ehF64aHRrKt566nrn_PYG9vAMsOLwOSgWhGxH6q5kpLQnWH8fowaIyY7KOJmIn8l5ypgUTqHlGyaIIw)
 - [QR-DQN (Quantile-Regression DQN)](https://arxiv.org/abs/1710.10044)
 - [REM (Random Ensemble Mixture)](https://proceedings.icml.cc/static/paper_files/icml/2020/5394-Paper.pdf)

The thesis PDF is located in the root of the project. If you intend to use the code or information from the thesis, make sure to cite.
```bibtex
@mastersthesis{
  author       = {Alexander Grimm}, 
  title        = {Model-Free Deep Reinforcement Learning for Interactive Recommender Systems: An Evaluation},
  school       = {Julius-Maximilians-University Würzburg, Germany},
  year         = 2020,
  address      = {https://github.com/almajo/drl4rec-thesis},
  month        = 6
}
```

## Requirements
The environment used can be built either with the requirements.txt file, but pytorch needs to be added. 
Otherwise the Dockerfile creates a full working docker container.

## Base implementation
The project's structure is based on and uses components from 
[p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch). 
The library is put under `drlap`.

However, most used functions and classes are re-implemented for efficiency and readability.

## Data Preprocessing

The used datasets can be downloaded from:

1. https://grouplens.org/datasets/movielens/
2. https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1

To preprocess the data do:
```bash
python3 data/movielens/prepare_movielens.py /path/to/ml/directory (same for 1 and 25M)
python3 data/split_prep_data.py /path/to/ml/directory 

python3 data/taobao/convert_events.py /path/to/taobao/directory
python3 data/split_prep_data.py /path/to/taoabo/directory 
```

the data is expected to be in the following directory structure:
```
data:
 |- taobao
 |- ml
   |- ml-1m
   |- ml-25m
```

The simulator is built with the script in `data/simulation/create_bpr_matrix.py `
and word2vec vectors with `data/word2vec/word2vec.py`. 

See the respective kube.yaml files for execution details.

## Experiment reproduction

The main script is in run/train_main.py and takes several arguments. 
A kubernetes version is also available in `scheduler.py` that automatically creates multiple jobs, 
one for each seed,dataset and agent respectively. 
All that needs to be done is to adapt information in `scheduler.py` and the yaml template can be found in run/kube.yaml.jinja2

Each ./scheduler can be replaced with python run/train_main.py, BUT it does only take _one_ seed,data and agent argument. 
Furthermore, --data must be ml/ml-1m ml/ml-25m respectively.

## Offline Evaluation

Batch-RL we always use word2vec embeddings and no pretraining (because everything is supervised). 
It is best to look at the arguments of the scheduler script. data, seed and agent can be a list of multiples.

A version with all arguments:
`./scheduler --agent gru qrdqn tpgr rem dueling news wolpertinger lird correction --data ml-1m taobao ml-25m 
--subdir batch-rl --name dir_name_in_subdir --word2vec 16 --batch --seed 0 1 2  --num_episodes 5000`

The essential keyword is `--batch`. 
This command would create 81 kubernetes jobs (9 agents x 3 datasets x 3 seeds).
A single run can also be specified with train_main directly

`python3 run/train_main.py --agent gru --data ml/ml-1m --subdir batch-rl --name dir_name_in_subdir 
--word2vec 16 --batch --seed 0  --num_episodes 5000`


The offline test is automatically executed after training, where it picks the best model on the valid set.

### Online Evaluation

For evaluation on the **simulator** do:

`python3 run/run_evaluation.py path_to_experiment_subdir bpr`

E.g. if in the offline evaluation everything was saved in the subdir/name batch-rl/dir_name_in_subdir, this would be 
the path and the best model seeds of each version is automatically found


### Static Baselines
For the baselines we need to 

`python run/run_static_baselines [ml/ml-1m or ml/ml-25m or taobao]`


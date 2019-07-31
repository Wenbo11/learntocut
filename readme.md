# Reinforcement Learning for Integer Programming: Learning to Cut 

This is the repo of NeurIPS 2019 submission - Reinforcement Learning for Integer Programming: Learning to Cut. 

## Dependencies
The code works in Python 2.7. You need to install Chainer, Cython, Gurobi (Gurobi-py).

## Build
You need to build all the C dependencies using Cython. In particular
```
cd libs/cwrapping/gurobipy
./build.sh
```
and also
```
cd es
python setup.py build_ext -i
```

## Installation
Install the current repo
```
pip install -e .
```

## Running
Before running, please do go to each main python file to check for missing hyper-parameters, loading directories, etc. Train a cut selection RL agent 
```
cd main/cuts
./train.sh
```
To run the RL agent on a B&C downstream task
```
cd main/bc
./run.sh
```


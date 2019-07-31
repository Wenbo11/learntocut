# Experiment scripts for training RL agent to select cuts

## General guideline

This repo contains a .py file and a .sh file. The .py file is the main program. The C-programming interface of Gurobi, when combined with python, seems to generate memory leaks, The leak builds up over time and eventually leads to a crash in the whole server. To bypass such an issue, we launch a .sh script that keeps launching new processes which run the python script. The python script periodically checks for the memory usage of the current program compared to the memory usage of the entire operating system. Once certain threshold is reached, the program will automatically terminate and launch a new python program. The old/new programs do proper book keeping so that the entire training loop does not break - it is as if we keep running the same python program.

## Instructions
The python file contains certain hyperparameters and directories which need to be specified by the user. Such parameters include the directories from which to load the original raw IP instances as well as the logging directory of the training loop. Please refer to the python file for more details.

It is our future work to tidy up the code base more in order to allow for more convenient parsing of command line arguments.

## Running
To train a RL agent 
```
./train.sh
```



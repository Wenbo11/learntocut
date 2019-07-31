# Experiment scripts for running trained RL agent to select cuts in a branch-and-bound setting

## General guideline

This repo contains a .py file and a .sh file. The .py file is the main program executed by the .sh file. The .sh file launches multiple processes in a sequence to test the performance of the trained RL agent under different settings, e.g. with different IP instances. The .sh file also allows for specifying certain hyper-parameters into the .py program in the form of command line. For example, the user is free to use other baseline algorithms for cut selection.

## Instructions
The python file contains certain hyperparameters and directories which need to be specified by the user. Such parameters include the directories from which to load the original raw IP instances as well as the loading directory of the trained policy. Please refer to the python file for more details.

It is our future work to tidy up the code base more in order to allow for more convenient parsing of command line arguments.

## Running
To run a RL agent 
```
./run.sh
```



# DRLFuzz

## Installation
DRLFuzz requires the following dependencies:
* numpy
* pytorch
* PyGame-Learning-Environment
* pygame
* scipy

## DRLFuzz experiments
```catcher```,```flaappy_bird``` and ```pong``` are the three experimental subjects in the papper. They all contain three files:```main.py```,```repair.py``` and ```verify.py```.
* ```main.py```: DRLFuzz script.
* ```repair.py```: repair script.
* ```verify.py```: verify the repair result script.

the ```result``` folder contains all experimental results.
* ```model```: contain unrepaired and repaired models.
* ```xxx.log```: experiment log.
* ```result_xxx.txt```: generated failed test cases.
* ```result.xlsx```: experimental analysis data.

## STARLA experiments

STARLA algorithm includes two parts: generating test cases and execute them. The results are regarded as the executed ones. Main code files are as follows:

* ```STARLA.py```: generate test cases.
* ```Execute_Results.py```: execute results in ```Results``` folder.
* ```random_test.py```: an example for using the environment. 

Note: ```STARLA.py``` and ```Execute_Results.py``` are modified from [STARLA](https://github.com/amirhosseinzlf/STARLA) under MIT license.

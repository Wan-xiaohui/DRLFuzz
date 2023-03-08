# DRLFuzz

## Installation
DRLFuzz requires the following dependencies:
* numpy
* pytorch
* PyGame-Learning-Environment
* pygame
* scipy

## Getting strated
```catcher```,```flaappy_bird``` and ```pong``` are the three experimental subjects in the papper. They all contain three files:```main.py```,```repair.py``` and ```verify.py```.
* ```main.py```: DRLFuzz script.
* ```repair.py```: repair script.
* ```verify.py```: verify the repair result script.

## Experimental results
the ```result``` folder contains all experimental results.
* ```model```: contain unrepaired and repaired models.
* ```xxx.log```: experiment log.
* ```result_xxx.txt```: generated failed test cases.
* ```result.xlsx```: experimental analysis data.

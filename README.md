# Decentralized-learning
Implementation of Reinforcement based Communication Topology Construction for Decentralized Learning with Non-IID Data
# Requirements:
* Python 3.7
* pytorch=1.2.0
# How to run?
```shell=
python run.py --config=config.json
```
* `--config` (`-c`): path to the configuration file to be used.

`config.json` files

We use a JSON file (i.e.,config.json) to manage the configuration parameters for decentralized learning simulation.
For training different dataset, you can modify the `model` in JSON file to "FashionMNIST" or "CIFAR-10".
If you want to adjust more detailed settings, you can modify the json file.

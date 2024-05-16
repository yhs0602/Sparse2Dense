# CraftGround-Baselines3
## Setup
[Headless Server Setup](https://yhs0602.github.io/CraftGround/headless)

## Experiment Setting
```shell
python cross_w2/experiments/sparse.py --goal 0 --port1 8000 --port2 8001 --device-id 0
```
- `--goal`: goal index to exclude during training
- `--port1`: port number for the train server
- `--port2`: port number for the eval server
- `--device-id`: cuda gpu id for training

## Experiment Results

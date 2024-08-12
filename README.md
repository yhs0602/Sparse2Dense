# Toddler 👶 inspired reward transition - Minecraft Experiment

This directory includes the source code and environments for running the toddler-inspired reward transition experiments
on Minecraft Toddler Playroom and Minecraft Cross Maze.

## Installation

Usage of Conda is recommended, and the current installation guide is centered around Conda.

1. Install the dependencies of the Minecraft environment and the Minecraft environment as below.
    ```shell
    sudo apt update
    sudo apt install libglew-dev cmake libpng-dev
   ```
2. Create a conda environment and activate it.
    ```shell
    conda create -n mincraft_maze python=3.9
    conda activate minecraft_maze
   ```
3. Prepare python 3.9 openjdk 21, and the required python packages.
    ```shell
    conda install -c conda-forge openjdk=21
    conda update -n base -c defaults conda
    python -m pip install -r requirements.txt
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
   if you want to run on headless server, follow the instructions of the next step.
4. The ways to run the minecraft server with 3d acceleration support, follow the
   steps [here](https://yhs0602.github.io/CraftGround/headless.html).

## Running experiments

There are two experiment packages in this repository, `room` and `cross_w2`. Both of them have the same structure, and
the structure is as follows:

- `room_env.py`: A handy function to create the Minecraft Toddler Playroom environment.
- `nbt`: A directory that contains the minecraft structure nbt files for the Minecraft Toddler Playroom environment.
- `wrappers`: A package that contains the wrappers for the Minecraft Toddler Playroom environment.
- `experiments`: A package that contains the experiments for the Minecraft Toddler Playroom environment.

The `experiments` package contains the following files:

- `sparse.py`: Only Sparse reward setting.
- `dense.py`: Only Dense reward setting.
- `transition.py`: Sparse to Dense reward transition setting.
- `d2s.py`: Dense to Sparse reward transition setting.

The `wrappers` package contains the following files:

- `room_episode_logger.py`: A wrapper that logs the episode information.
- `room_reach_check_log_wrapper.py`: A wrapper that checks if the agent reaches the goal, and logs the information.
- `room_goal_spawn_setup_wrapper.py`: A wrapper that sets up the goal and agent's spawn position.
- `room_dense_wrapper.py`: A wrapper that provides the dense reward signal.

To run the experiments, you can run the following command in the root directory:

```shell
export PYTHONPATH=.
python room/experiments/sparse.py  --port1 31001 --device-id 0 --extended
python room/experiments/dense.py  --port1 31013 --device-id 0 --extended 
python room/experiments/transition.py  --port1 33001 --device-id 0 --transition-timing 3000000 --extended
python room/experiments/d2s.py  --port1 31007 --device-id 0 --transition-timing 3000000 --extended   
```

The options are as follows:

- `port1`: The port number of the Minecraft environment server. Must not overlap.
- `device-id`: The device id of the GPU.
- `transition-timing`: The timing of the transition, in steps. The reward strategy changes after this number of steps,
  at the start of the following episode.
- `extended`: The flag to use the extended version of the environment. If this flag is set, the walls of the room are
  extended by a block.

For the cross maze settings, you can run the following command in the root directory:

```shell
export PYTHONPATH=.
python cross_w2/experiments/sparse.py --goal 2 --port1 11031 --port2 11032 --device-id 1
python cross_w2/experiments/dense.py --goal 2 --port1 11035 --port2 11036 --device-id 1
python cross_w2/experiments/transition.py --goal 2 --port1 11041 --port2 11042 --device-id 1 --transition-timing 3000000
python cross_w2/experiments/d2s.py --goal 2 --port1 11017 --port2 11018 --device-id 2 --transition-timing 3000000
```

The options are as follows:

- `goal`: The goal number of the cross maze. (0, 1, 2)
- `port1`: The port number of the Minecraft environment server for training. Must not overlap.
- `port2`: The port number of the Minecraft environment server for evaluation. Must not overlap.
- `device-id`: The device id of the GPU.
- `transition-timing`: The timing of the transition, in steps. The reward strategy changes after this number of steps,
  at the start of the following episode.

## Other experiments

You may try other experiments settings in the following directories also:

- `cross`: The cross maze environment, but the width of the corridor is 1.
- `h_maze`: The H-shaped maze environment.
- `h_maze_w2`: The H-shaped maze environment, but the width of the corridor is 2.

# Post processing scripts

To automatically generate figures and tables, we used the post-processing scripts in the `post_processing` directory.
This directory primarily contains the following files:

- `traj_lens.py`: A script calculates all the trajectory lengths of the experiments.
- `merge_trajectories.py`: A script that merges the files containing the length of the trajectories, generated
  by `traj_lens.py`, into a single csv file.
- `select_one_trajectory_and_inverse_action.py`: A script that selects one trajectory and generates a json file that
  contains the inversed actions.
- `room_movie.py`: A script that generates a trajectory video of the Minecraft Toddler Playroom environment.
- `room_bg.png`: A background image of the Minecraft Toddler Playroom environment.
- `cross_bg.png`: A background image of the Minecraft Cross Maze environment.
- `overlay_all_trajectories.py`: A script that overlays all the trajectories of the Minecraft Toddler Playroom
  environment, for calibration with the background image `room_bg.png`.
- `all_trajectories.png`: A image generated by `overlay_all_trajectories.py`.
- `movie.py`: Implementation of the trajectory video generation.
- `get_metrics.py`:
- `gen_csv_from_json.py`: A script that generates a csv file from the json analysis data, for easier analysis.
- `find_run.py`: A handy script for finding a run with 3111 steps.
- `draw_figures.py`: A script that generates a figure of
- `draw_feature_figure.py`: A script that generates the figure
- `download_all_run_data.py`: A script that downloads all the run data from the selected groups, for analysis.
- `change_step_metric.py`: A script that changes the step metric of the trajectory data.
- `calculate_mean.py`:
- `alpha_trajectory.py`: A script that generates a trajectory image from run data.
- `add_backgrounds.py`: A script that overlays the trajectory images on the background image, such as `room_bg.png`
  and `cross_bg.png`.

## Miscellaneous

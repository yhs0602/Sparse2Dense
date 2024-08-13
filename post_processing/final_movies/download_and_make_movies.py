import os
import wandb
from wandb.apis.public import Run, File

from post_processing.final_movies.movie_without_bg import create_trajectory_movie
from wandb_envs import WANDB_ENTITY, WANDB_PROJECT

current_file_dir = os.path.dirname(os.path.abspath(__file__))
cross_dir = os.path.join(current_file_dir, "cross")
room_dir = os.path.join(current_file_dir, "room")


def main():
    cross_runs = {
        "sparse": [
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/ludeknjz",
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/3tvmsrec",
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/gvkoowlj",
        ],
        "dense": [
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/xy8clqb9",
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/0itxorpm",
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/9gcgf7pj",
        ],
        "d2s": [
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/b596dbfe",
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/js96zfgu",
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/5iv05pfx",
        ],
        "s2d": [
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/jp71xuyn",
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/nk54yfnh",
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/83r3f8hi",
        ],
    }
    room_runs = {
        "sparse": f"{WANDB_ENTITY}/{WANDB_PROJECT}/3l88j5yt",
        "dense": f"{WANDB_ENTITY}/{WANDB_PROJECT}/frc5sa1s",
        "d2s": f"{WANDB_ENTITY}/{WANDB_PROJECT}/7z2q9oli",
        "s2d": f"{WANDB_ENTITY}/{WANDB_PROJECT}/fgt7xdhv",
    }

    api = wandb.Api(timeout=120)
    for reward in cross_runs:
        for i, goal_run in enumerate(cross_runs[reward]):
            cross_goal_dir = os.path.join(cross_dir, str(i))
            run: Run = api.run(goal_run)
            os.makedirs(cross_goal_dir, exist_ok=True)
            # First get the trajectory and episode number
            print(run.summary.keys())
            positions = run.summary["eval_episode/positions"]
            videos = run.summary["videos"]
            last_video_path = videos["path"]
            # download the movie
            file: File = run.file(last_video_path)
            file.download(root=current_file_dir, api=api, exist_ok=True)
            target_name = os.path.join(cross_goal_dir, f"{reward}_ego.mp4")
            os.rename(file.name, os.path.join(cross_goal_dir, target_name))
            traj_file_name = os.path.join(cross_goal_dir, f"{reward}_traj.png")
            eval_episode = run.summary["eval_episode"]
            goals = run.summary["eval_episode/goal"]
            # make movie from the data
            min_x = 1
            max_x = 14
            min_z = 0
            max_z = 13
            create_trajectory_movie(
                positions, traj_file_name, goals, min_x, max_x, min_z, max_z
            )
            break


if __name__ == "__main__":
    main()

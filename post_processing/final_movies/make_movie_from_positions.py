import json
import os

from post_processing.final_movies.movie_without_bg import create_trajectory_movie

current_folder = os.path.dirname(os.path.abspath(__file__))
room_media_folder = os.path.join(current_folder, "room_media")


def main():
    # for each json in room_media_folder
    for json_file in os.listdir(room_media_folder):
        if not json_file.endswith(".json"):
            continue
        if "cross" in json_file:
            min_x = 1
            max_x = 14
            min_z = 0
            max_z = 13
        elif "room" in json_file:
            min_x = 0
            max_x = 12
            min_z = 0
            max_z = 19
        else:
            print(f"Unknown group: {json_file}")
            continue
        # load the json
        content = json.load(open(os.path.join(room_media_folder, json_file)))
        # get the positions
        positions = content["positions"]
        # get the goals
        goals = content["goal"]
        # create the trajectory movie
        traj_file_name = os.path.join(room_media_folder, f"{json_file}.mp4")
        create_trajectory_movie(
            positions, traj_file_name, goals, min_x, max_x, min_z, max_z
        )


if __name__ == "__main__":
    main()

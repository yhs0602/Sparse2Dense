import subprocess

import pygame
from pygame import Rect

from post_processing.alpha_trajectory import mc_coord_to_image_coord, get_color


def create_trajectory_movie(
    positions, filename, goals, min_x, max_x, min_z, max_z, frame_rate=20
):
    pygame.init()
    cell_size = 20
    agent_size_in_cell = 0.3
    goal_size_in_cell = 0.1
    # get grid size
    # for Mc: traverse from (min_x, min_z) Left bottom to (max_x, max_z) Right top
    # Image: (min_z, max_x - min_x) ~ (max_z, 0) to navigate around
    grid_x_length = max_z - min_z
    grid_z_length = max_x - min_x

    print(f"{grid_x_length} x {grid_z_length}")
    width = grid_x_length * cell_size + 2
    height = grid_z_length * cell_size + 2
    screen = pygame.display.set_mode((width, height))

    command = [
        "ffmpeg",
        "-y",  # Replace the older file
        "-f",
        "rawvideo",  # Input format
        "-vcodec",
        "rawvideo",  # Input codec
        "-s",
        f"{width}x{height}",  # Input resolution
        "-pix_fmt",
        "rgb24",  # Input pixel format
        "-r",
        str(frame_rate),  # Input framerate
        "-i",
        "-",  # Input from stdin
        "-an",  # No audio
        "-vcodec",
        "mpeg4",  # Output codec
        "-b:v",
        "5000k",  # Set bitrate
        filename,
    ]

    # Start the FFmpeg process
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    # Goal Draw
    # Goal squeeze
    goals = [(goal[0], goal[2]) if len(goal) == 3 else goal for goal in goals]

    image_goal_coords = [
        mc_coord_to_image_coord(goal[0], goal[1], max_x, max_z) for goal in goals
    ]
    image_positions = [
        mc_coord_to_image_coord(pos[0], pos[2], max_x, max_z) for pos in positions
    ]
    print(f"goal: {goals} -> {image_goal_coords}")
    for goal in image_goal_coords:
        goal_coord_in_image = (goal[0] * cell_size, goal[1] * cell_size)
        pygame.draw.ellipse(
            screen,
            "green",
            Rect(
                goal_coord_in_image[0],
                goal_coord_in_image[1],
                10,
                10,
            ),
        )
    # The alpha value is 255 for the most recent and 0 for the oldest.
    last_index = len(positions) - 1
    for i in range(len(positions) - 1):
        pygame.draw.line(
            screen,
            get_color(i, last_index, (0, 255, 0), 120),  # (0, 255, 255, alpha),
            [(image_positions[i][0]) * cell_size, (image_positions[i][1]) * cell_size],
            [
                (image_positions[i + 1][0]) * cell_size,
                (image_positions[i + 1][1]) * cell_size,
            ],
            width=int(cell_size * agent_size_in_cell),
        )
        # 끝점 Draw
        pygame.draw.ellipse(
            screen,
            "red",
            Rect(
                int((image_positions[-1][0] - agent_size_in_cell) * cell_size),
                int((image_positions[-1][1] - agent_size_in_cell) * cell_size),
                agent_size_in_cell * cell_size,
                agent_size_in_cell * cell_size,
            ),
        )
        # 시작점 Draw
        pygame.draw.ellipse(
            screen,
            "blue",
            Rect(
                int((image_positions[0][0] - agent_size_in_cell) * cell_size),
                int((image_positions[0][1] - agent_size_in_cell) * cell_size),
                agent_size_in_cell * cell_size,
                agent_size_in_cell * cell_size,
            ),
        )
        frame = pygame.surfarray.array3d(screen)
        frame = frame.swapaxes(0, 1)  # Changing axes between Pygame and regular image formats
        process.stdin.write(frame.tobytes())  # Convert frame data to bytes and send to FFmpeg

        pygame.display.flip()
    process.stdin.close()
    process.wait()
    pygame.quit()

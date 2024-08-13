import subprocess

import pygame
import tqdm
import wandb

from wandb_envs import WANDB_PROJECT

# 21 x 16 Maze
maze_str = [
    "oooxxxxxxxxxxooo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxooooooooooooxo",
    "oxxxxxxxxxxxxxxo",  # Aisle
    "oxooooooooooooxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oxoxxxxxxxxxxoxo",
    "oooxxxxxxxxxxooo",
]
assert len(maze_str) == 21


# start point: 3 1 1 > 0 0 0


def create_video_from_positions(
    maze,
    x_offset,
    y_offset,
    positions,
    episode_id,
    goal=(0, 0),
    block_size=10,
    frame_rate=20,  # 20 TPS
):
    pygame.init()
    maze_size_w, maze_size_h = len(maze[0]), len(maze)
    width, height = (maze_size_w + abs(x_offset)) * block_size, (
        maze_size_h + abs(y_offset)
    ) * block_size
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    video_filename = f"episode_{episode_id}.mp4"
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
        video_filename,
    ]

    # Start FFmpeg process
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    pos0 = positions[0]
    dimension = len(pos0)

    # cache maze surface and goal
    background = pygame.Surface((width, height))
    background.fill((255, 255, 255))
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == "o":
                pygame.draw.rect(
                    background,
                    (0, 0, 0),
                    (
                        (x + x_offset) * block_size,
                        (y + y_offset) * block_size,
                        block_size,
                        block_size,
                    ),
                )
    # Goal Draw
    goal_x, goal_y = goal
    pygame.draw.circle(
        background,
        (0, 255, 0),
        (
            goal_x * block_size + int(block_size / 2),
            goal_y * block_size + int(block_size / 2),
        ),
        int(block_size / 2),
    )

    # Based on Agent positions generate frames
    for position in tqdm.tqdm(positions):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        screen.blit(background, (0, 0))  # Background Draw
        # TODO: Cache maze surface
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                color = (0, 0, 0) if cell == "o" else (255, 255, 255)
                pygame.draw.rect(
                    screen,
                    color,
                    (
                        (x + x_offset) * block_size,
                        (y + y_offset) * block_size,
                        block_size,
                        block_size,
                    ),
                )
        if dimension == 3:
            y, z, x = position
            yaw = 0
        elif dimension == 4:
            y, z, x, yaw = position
        else:
            raise ValueError(f"Invalid dimension {dimension}")

        agent_size = int(block_size * 0.6)  # (hitbox = 0.6 x 1.8 x 0.6)
        agent_radius = int(agent_size / 2)
        agent_dx = int(agent_radius * 0.8660254038)  # sqrt(3)/2
        agent_dy = int(agent_radius * 0.5)
        # draw triangle based on yaw
        yaw = int(yaw / 90) % 4
        agent_color = (255, 0, 0)
        if yaw == 0:
            pygame.draw.polygon(
                screen,
                agent_color,
                [
                    (int(x * block_size) - agent_dx, int(y * block_size) - agent_dy),
                    (int(x * block_size), int(y * block_size) + agent_radius),
                    (int(x * block_size) + agent_dx, int(y * block_size) - agent_dy),
                ],
            )
        elif yaw == 1:
            pygame.draw.polygon(
                screen,
                agent_color,
                [
                    (int(x * block_size) - agent_dy, int(y * block_size) - agent_dx),
                    (int(x * block_size) + agent_radius, int(y * block_size)),
                    (int(x * block_size) - agent_dy, int(y * block_size) + agent_dx),
                ],
            )
        elif yaw == 2:
            pygame.draw.polygon(
                screen,
                agent_color,
                [
                    (int(x * block_size) + agent_dx, int(y * block_size) + agent_dy),
                    (int(x * block_size), int(y * block_size) - agent_radius),
                    (int(x * block_size) - agent_dx, int(y * block_size) + agent_dy),
                ],
            )
        elif yaw == 3:
            pygame.draw.polygon(
                screen,
                agent_color,
                [
                    (int(x * block_size) + agent_dy, int(y * block_size) + agent_dx),
                    (int(x * block_size) - agent_radius, int(y * block_size)),
                    (int(x * block_size) + agent_dy, int(y * block_size) - agent_dx),
                ],
            )
        else:
            pygame.draw.circle(
                screen,
                (255, 0, 0),
                (
                    int(x * block_size) + agent_radius,
                    int(y * block_size) + agent_radius,
                ),
                agent_radius,
            )  # Agent Draw

        # Pipe the frame to ffmpeg
        frame = pygame.surfarray.array3d(screen)
        frame = frame.swapaxes(
            0, 1
        )  # Changing axes between Pygame and regular image formats
        process.stdin.write(
            frame.tobytes()
        )  # Convert frame data to bytes and send to FFmpeg

        pygame.display.flip()
        # clock.tick(frame_rate)  # Setting the frame rate

    # Cleaning up FFmpeg and Pygame
    process.stdin.close()
    process.wait()
    pygame.quit()


def make_movie():
    # Initialize W&B API
    api = wandb.Api(timeout=30)

    # Select the project and run
    project_name = WANDB_PROJECT
    run_id = "zypbugn5"
    run = api.run(f"{project_name}/{run_id}")

    # Get the log data
    data = run.history(keys=["episode/positions"], pandas=False)
    # Generate videos for each episode
    n = 0
    for episode_id, episode_data in enumerate(data):
        positions = episode_data["episode/positions"]
        create_video_from_positions(maze_str, 0, 2, positions, episode_id)
        n += 1
        if n >= 3:
            break


if __name__ == "__main__":
    make_movie()

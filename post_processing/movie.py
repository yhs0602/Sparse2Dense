import colorsys
import subprocess
from collections import deque
from typing import List, Tuple

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
    "oxxxxxxxxxxxxxxo",  # 통로
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
    video_filename,
    episode: int = 0,
    reached_goal: int = 0,
    goals: List[Tuple[int, int]] = [(0, 0)],
    block_size=10,
    frame_rate=20,  # 20 TPS
    palette=None,
):
    if palette is None:
        palette = {
            "o": (0, 0, 0),
            "default": (255, 255, 255),
        }
    pygame.init()
    maze_size_w, maze_size_h = len(maze[0]), len(maze)
    width, height = (maze_size_w + abs(x_offset)) * block_size, (
        maze_size_h + abs(y_offset)
    ) * block_size
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
        video_filename,
    ]

    # FFmpeg 프로세스 시작
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    pos0 = positions[0]
    dimension = len(pos0)

    # cache maze surface and goal
    background = pygame.Surface((width, height))
    default_color = palette["default"]
    background.fill(default_color)
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            color = palette.get(cell, None)
            if color is not None:
                pygame.draw.rect(
                    background,
                    color,
                    (
                        (x + x_offset) * block_size,
                        (y + y_offset) * block_size,
                        block_size,
                        block_size,
                    ),
                )
    # Goal Draw
    font = pygame.font.Font(None, 18)
    for idx, goal in enumerate(goals):
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
        # Goal 좌표 출력
        text = font.render(f"{goal[0], goal[1]}", True, (0, 255, 0))
        background.blit(text, (idx * 30, 0))
    # episode 출력
    text = font.render(f"Ep.{episode}({reached_goal})", True, (0, 0, 255))
    background.blit(text, (width - 60, 0))
    # Agent 위치를 기반으로 프레임 생성
    last_n_poses = deque(maxlen=len(positions))
    for time, position in tqdm.tqdm(enumerate(positions)):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        screen.blit(background, (0, 0))  # 배경 Draw
        pos_txt = font.render(
            f"{int(position[0]), int(position[2])}", True, (255, 0, 0)
        )
        screen.blit(pos_txt, (0, 20))
        time_txt = font.render(f"T:{time}", True, (255, 0, 255))
        screen.blit(time_txt, (width - 40, 20))
        if dimension == 3:
            x, z, y = position
            yaw = 0
        elif dimension == 4:
            x, z, y, yaw = position
        else:
            raise ValueError(f"Invalid dimension {dimension}")

        agent_size = int(block_size * 0.6)  # (hitbox = 0.6 x 1.8 x 0.6)
        agent_radius = int(agent_size / 2)
        agent_dx = int(agent_radius * 0.8660254038)  # sqrt(3)/2
        agent_dy = int(agent_radius * 0.5)
        # draw triangle based on yaw
        yaw = int(yaw / 90) % 4
        agent_color = (255, 0, 0)
        agent_realx = int(x * block_size)
        agent_realy = int(y * block_size)
        last_n_poses.append((agent_realx, agent_realy))
        if yaw == 0:
            pygame.draw.polygon(
                screen,
                agent_color,
                [
                    (agent_realx - agent_dx, int(y * block_size) - agent_dy),
                    (agent_realx, int(y * block_size) + agent_radius),
                    (agent_realx + agent_dx, int(y * block_size) - agent_dy),
                ],
            )
        elif yaw == 1:
            pygame.draw.polygon(
                screen,
                agent_color,
                [
                    (agent_realx - agent_dy, int(y * block_size) - agent_dx),
                    (agent_realx + agent_radius, int(y * block_size)),
                    (agent_realx - agent_dy, int(y * block_size) + agent_dx),
                ],
            )
        elif yaw == 2:
            pygame.draw.polygon(
                screen,
                agent_color,
                [
                    (agent_realx + agent_dx, int(y * block_size) + agent_dy),
                    (agent_realx, int(y * block_size) - agent_radius),
                    (agent_realx - agent_dx, int(y * block_size) + agent_dy),
                ],
            )
        elif yaw == 3:
            pygame.draw.polygon(
                screen,
                agent_color,
                [
                    (agent_realx + agent_dy, int(y * block_size) + agent_dx),
                    (agent_realx - agent_radius, int(y * block_size)),
                    (agent_realx + agent_dy, int(y * block_size) - agent_dx),
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
        # 궤적 Draw, draw_lines
        if len(last_n_poses) > 1:
            for i in range(1, len(last_n_poses)):
                start_pos = last_n_poses[i - 1]
                end_pos = last_n_poses[i]
                # hsv version
                if False:
                    hue = i / len(last_n_poses)  # hue 값은 0에서 1 사이
                    color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                    color = tuple(int(c * 255) for c in color)  # RGB로 변환
                # intensity version
                else:
                    # The recent, the more intense, exponentially
                    recensity = (len(last_n_poses) - i) / len(last_n_poses)
                    color_intensity = min(255 * (1 - 0.3 ** (recensity * 15)), 230)
                    # Naive version: color_intensity = max(int(255 * i / len(last_n_poses)), 40)
                    color = (
                        color_intensity,
                        color_intensity,
                        color_intensity,
                    )
                pygame.draw.line(screen, color, start_pos, end_pos, 2)

        # 프레임을 FFmpeg로 파이프
        frame = pygame.surfarray.array3d(screen)
        frame = frame.swapaxes(0, 1)  # Pygame과 일반 이미지 포맷 간의 축 변경
        process.stdin.write(frame.tobytes())  # 프레임 데이터를 바이트로 변환 후 FFmpeg에 전송

        pygame.display.flip()
        # clock.tick(frame_rate)  # 프레임 레이트 설정

    # FFmpeg와 Pygame 정리
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

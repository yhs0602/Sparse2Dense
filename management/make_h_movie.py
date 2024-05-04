import subprocess

import pygame
import tqdm
import wandb

# 21 x 16 미로
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
        "-y",  # 기존 파일 덮어쓰기
        "-f",
        "rawvideo",  # 입력 형식
        "-vcodec",
        "rawvideo",  # 입력 코덱
        "-s",
        f"{width}x{height}",  # 입력 크기
        "-pix_fmt",
        "rgb24",  # 입력 픽셀 포맷
        "-r",
        str(frame_rate),  # 입력 프레임레이트
        "-i",
        "-",  # stdin을 통해 입력
        "-an",  # 오디오 무시
        "-vcodec",
        "mpeg4",  # 출력 코덱
        "-b:v",
        "5000k",  # 비트레이트 설정
        video_filename,
    ]

    # FFmpeg 프로세스 시작
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
    # Goal 그리기
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

    # 에이전트 위치를 기반으로 프레임 생성
    for position in tqdm.tqdm(positions):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        screen.blit(background, (0, 0))  # 배경 그리기
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
            )  # 에이전트 그리기

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
    # W&B API 초기화
    api = wandb.Api(timeout=30)

    # 특정 프로젝트와 run ID 지정
    project_name = "craftground-sb3"
    run_id = "zypbugn5"
    run = api.run(f"{project_name}/{run_id}")

    # 로그 데이터 가져오기
    data = run.history(keys=["episode/positions"], pandas=False)
    # 각 에피소드별로 동영상 생성
    n = 0
    for episode_id, episode_data in enumerate(data):
        positions = episode_data["episode/positions"]
        create_video_from_positions(maze_str, 0, 2, positions, episode_id)
        n += 1
        if n >= 3:
            break


if __name__ == "__main__":
    make_movie()

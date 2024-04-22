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
    "oooxxxxxxxxxxooo",
]
assert len(maze_str) == 21


# start point: 3 1 1 > 0 0 0


def create_video_from_positions(maze, positions, episode_id, frame_rate=1000):
    pygame.init()
    width, height = 160, 210
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

    # 에이전트 위치를 기반으로 프레임 생성
    for position in tqdm.tqdm(positions):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        screen.fill((255, 255, 255))  # 배경색 설정
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                color = (0, 0, 0) if cell == "o" else (0, 255, 0)
                pygame.draw.rect(screen, color, (x * 10, y * 10, 10, 10))
        y, z, x = position
        pygame.draw.circle(
            screen, (255, 0, 0), (int(x) * 10 + 5, int(y) * 10 + 5), 5
        )  # 에이전트 그리기

        # 프레임을 FFmpeg로 파이프
        frame = pygame.surfarray.array3d(screen)
        frame = frame.swapaxes(0, 1)  # Pygame과 일반 이미지 포맷 간의 축 변경
        process.stdin.write(frame.tobytes())  # 프레임 데이터를 바이트로 변환 후 FFmpeg에 전송

        pygame.display.flip()
        clock.tick(frame_rate)  # 프레임 레이트 설정

    # FFmpeg와 Pygame 정리
    process.stdin.close()
    process.wait()
    pygame.quit()


def make_movie():
    # W&B API 초기화
    api = wandb.Api()

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
        create_video_from_positions(maze_str, positions, episode_id)
        n += 1
        if n >= 3:
            break


if __name__ == "__main__":
    make_movie()

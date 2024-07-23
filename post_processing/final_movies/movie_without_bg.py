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
    # grid 사이즈 구함
    # Mc: (min_x, min_z) Left bottom ~ (max_x, max_z) Right top 에서 돌아다니면
    # Image에서는 (min_z, max_x - min_x) ~ (max_z, 0) 에서 돌아다님
    grid_x_length = max_z - min_z
    grid_z_length = max_x - min_x

    print(f"{grid_x_length} x {grid_z_length}")
    width = grid_x_length * cell_size + 2
    height = grid_z_length * cell_size + 2
    screen = pygame.display.set_mode((width, height))

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
        filename,
    ]

    # FFmpeg 프로세스 시작
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    # Goal 그리기
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
    # 알파값은 최근일수록 255, 이전일수록 0
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
        # 끝점 그리기
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
        # 시작점 그리기
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
        frame = frame.swapaxes(0, 1)  # Pygame과 일반 이미지 포맷 간의 축 변경
        process.stdin.write(frame.tobytes())  # 프레임 데이터를 바이트로 변환 후 FFmpeg에 전송

        pygame.display.flip()
    process.stdin.close()
    process.wait()
    pygame.quit()

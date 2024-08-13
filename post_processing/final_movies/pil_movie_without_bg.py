# 13 x 13
import colorsys
import subprocess
from collections import deque
from typing import Tuple

import numpy as np
from PIL import ImageDraw, Image, ImageChops
from PIL.Image import Resampling


# 0, 0 -> 12, 19
# 6.0, ~, 9.5


def mc_coord_to_image_coord(x, z, max_x, max_z) -> Tuple[float, float]:
    return z, max_x - x


def hsl_to_rgb(h, s, l):
    return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h, l, s))


def get_color(
    step, total_steps, base_color: Tuple[int, int, int], alpha: int
) -> Tuple[int, int, int, int]:
    # base_color는 (R, G, B) 형식
    h, l, s = colorsys.rgb_to_hls(
        base_color[0] / 255, base_color[1] / 255, base_color[2] / 255
    )
    lightness = min(step / total_steps, 0.7)
    r, g, b = hsl_to_rgb(h, s, lightness)
    return r, g, b, alpha


def create_trajectory_video_using_pil(
    positions,
    filename,
    goals,
    min_x,
    max_x,
    min_z,
    max_z,
    background_image_path,
    scale_factor_hw,
    frame_rate=20,  # 20 TPS
):
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
        "rgba",  # Input pixel format
        "-r",
        str(frame_rate),  # Input framerate
        "-i",
        "-",  # Input from stdin
        "-an",  # No audio
        "-c:v",
        "libvpx-vp9",  # Output codec that supports alpha channel
        # "-vcodec",
        # "libvpx-vp9",  # Output codec that supports alpha channel
        "-pix_fmt",
        "yuva420p",  # Output pixel format with alpha channel
        "-b:v",
        "5000k",  # Set bitrate
        filename,
    ]
    # Start the FFmpeg process
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    # Prepare a background image
    background_image = Image.open(background_image_path)
    background_image = background_image.resize((width, height))
    new_size = (
        int(background_image.width * scale_factor_hw[1]),
        int(background_image.height * scale_factor_hw[0]),
    )

    img = Image.new(
        "RGBA",
        (width, height),
        color=(0, 0, 0, 0),
    )
    img.putalpha(0)
    # Grid Draw
    draw = ImageDraw.Draw(img, "RGBA")

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
        draw.ellipse(
            [
                (goal[0] - goal_size_in_cell) * cell_size,
                (goal[1] - goal_size_in_cell) * cell_size,
                (goal[0] + goal_size_in_cell) * cell_size,
                (goal[1] + goal_size_in_cell) * cell_size,
            ],
            fill="green",
        )
    # The alpha value is 255 for the most recent and 0 for the oldest.
    last_index = len(positions) - 1
    tmp_imgs = deque(maxlen=3)
    for i in range(len(positions) - 1):
        tmp_img = Image.new(
            "RGBA",
            (width, height),
            color=(0, 0, 0, 0),
        )
        tmp_img.putalpha(0)
        tmp_draw = ImageDraw.Draw(tmp_img, "RGBA")
        tmp_draw.line(
            [
                (image_positions[i][0]) * cell_size,
                (image_positions[i][1]) * cell_size,
                (image_positions[i + 1][0]) * cell_size,
                (image_positions[i + 1][1]) * cell_size,
            ],
            fill=get_color(i, last_index, (0, 255, 0), 120),  # (0, 255, 255, alpha),
            width=int(cell_size * agent_size_in_cell),
            joint="curve",
        )
        tmp_imgs.append(tmp_img)
        # AB + BC - should be (AB ^ BC).
        if len(tmp_imgs) == 2:
            newer_img = tmp_imgs[1]
            older_img = tmp_imgs[0]
            tmp_img = ImageChops.subtract(newer_img, older_img)
        elif len(tmp_imgs) == 3:
            tmp_img = ImageChops.subtract(
                tmp_imgs[2], ImageChops.add(tmp_imgs[0], tmp_imgs[1])
            )
        else:
            tmp_img = tmp_imgs[0]
        img = Image.alpha_composite(img, tmp_img)
        del tmp_img, tmp_draw
        # 끝점 Draw
        draw = ImageDraw.Draw(img, "RGBA")
        draw.ellipse(
            [
                (image_positions[-1][0] - agent_size_in_cell) * cell_size,
                (image_positions[-1][1] - agent_size_in_cell) * cell_size,
                (image_positions[-1][0] + agent_size_in_cell) * cell_size,
                (image_positions[-1][1] + agent_size_in_cell) * cell_size,
            ],
            fill="red",
        )
        # 시작점 Draw
        draw.ellipse(
            [
                (image_positions[0][0] - agent_size_in_cell) * cell_size,
                (image_positions[0][1] - agent_size_in_cell) * cell_size,
                (image_positions[0][0] + agent_size_in_cell) * cell_size,
                (image_positions[0][1] + agent_size_in_cell) * cell_size,
            ],
            fill="blue",
        )

        trajectory_image = img.copy()
        trajectory_image = trajectory_image.resize(
            new_size, resample=Resampling.BICUBIC
        )
        # paste to match left bottom
        position = (
            0,
            background_image.height - trajectory_image.height,
        )
        background_image_copy = background_image.copy()
        background_image_copy.paste(trajectory_image, position, trajectory_image)
        frame = np.array(background_image_copy)  # Convert PIL image to numpy array
        # print(frame.shape)
        process.stdin.write(frame.tobytes())  # Write frame data to FFmpeg
    # FFmpeg와 Pygame 정리
    process.stdin.close()
    process.wait()

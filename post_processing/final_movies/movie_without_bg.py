from collections import deque

from PIL import ImageDraw, Image, ImageChops

from post_processing.alpha_trajectory import mc_coord_to_image_coord, get_color


def create_trajectory_movie(positions, filename, goals, min_x, max_x, min_z, max_z):
    cell_size = 20
    agent_size_in_cell = 0.3
    goal_size_in_cell = 0.1
    # grid 사이즈 구함
    # Mc: (min_x, min_z) Left bottom ~ (max_x, max_z) Right top 에서 돌아다니면
    # Image에서는 (min_z, max_x - min_x) ~ (max_z, 0) 에서 돌아다님
    grid_x_length = max_z - min_z
    grid_z_length = max_x - min_x

    print(f"{grid_x_length} x {grid_z_length}")

    # PIL로 Image 생성
    img = Image.new(
        "RGBA",
        (grid_x_length * cell_size + 1, grid_z_length * cell_size + 1),
        color="white",
    )
    img.putalpha(0)
    # Grid 그리기
    draw = ImageDraw.Draw(img, "RGBA")

    # for x in range(grid_x_length + 1):
    #     draw.line(
    #         [(x * cell_size, 0), (x * cell_size, grid_z_length * cell_size)],
    #         fill="black",
    #         width=1,
    #     )
    # for z in range(grid_z_length + 1):
    #     draw.line(
    #         [(0, z * cell_size), (grid_x_length * cell_size, z * cell_size)],
    #         fill="black",
    #         width=1,
    #     )
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
        draw.ellipse(
            [
                (goal[0] - goal_size_in_cell) * cell_size,
                (goal[1] - goal_size_in_cell) * cell_size,
                (goal[0] + goal_size_in_cell) * cell_size,
                (goal[1] + goal_size_in_cell) * cell_size,
            ],
            fill="green",
        )
    # Trajectory 그리기
    # for pos in positions:
    #     tmp_img = Image.new(
    #         "RGBA",
    #         (grid_x_length * cell_size + 1, grid_z_length * cell_size + 1),
    #     )
    #     tmp_img.putalpha(0)
    #     tmp_draw = ImageDraw.Draw(tmp_img, "RGBA")
    #     tmp_draw.ellipse(
    #         [
    #             (pos[0] - grid_min_x - agent_size_in_cell) * cell_size,
    #             (pos[1] - grid_min_z - agent_size_in_cell) * cell_size,
    #             (pos[0] - grid_min_x + agent_size_in_cell) * cell_size,
    #             (pos[1] - grid_min_z + agent_size_in_cell) * cell_size,
    #         ],
    #         fill=(117, 0, 0, 10),
    #     )
    #     img = Image.alpha_composite(img, tmp_img)
    #     del tmp_img, tmp_draw
    # 알파값은 최근일수록 255, 이전일수록 0
    last_index = len(positions) - 1
    tmp_imgs = deque(maxlen=3)
    for i in range(len(positions) - 1):
        # alpha = max(int(255 * i / last_index), 20)
        # k = 3
        # alpha = min(
        #     max(int(255 * (math.exp(k * i / last_index) - 1) / (math.exp(1) - 1)), 20),
        #     255,
        # )
        tmp_img = Image.new(
            "RGBA",
            (grid_x_length * cell_size + 1, grid_z_length * cell_size + 1),
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
        # AB + BC - (AB ^ BC) 해야 함.
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
    # 끝점 그리기
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
    # 시작점 그리기
    draw.ellipse(
        [
            (image_positions[0][0] - agent_size_in_cell) * cell_size,
            (image_positions[0][1] - agent_size_in_cell) * cell_size,
            (image_positions[0][0] + agent_size_in_cell) * cell_size,
            (image_positions[0][1] + agent_size_in_cell) * cell_size,
        ],
        fill="blue",
    )
    img.save(filename)

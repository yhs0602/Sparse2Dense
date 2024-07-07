import os
from typing import Tuple

from PIL import Image
from PIL.Image import Resampling


# Cross:
# min_x = 1
#             max_x = 14
#             min_z = 0
#             max_z = 13
# ratio = 13 / 15

# room:
# min_x = 0
#             max_x = 12
#             min_z = 0
#             max_z = 19
# ratio = 12 / 14


def overlay_images(
    trajectory_image_path,
    background_image_path,
    output_image_path,
    scale_factor_hw: Tuple[float, float],
):
    trajectory_image = Image.open(trajectory_image_path)
    background_image = Image.open(background_image_path)
    background_image = background_image.resize(trajectory_image.size)

    new_size = (
        int(background_image.width * scale_factor_hw[1]),
        int(background_image.height * scale_factor_hw[0]),
    )
    trajectory_image = trajectory_image.resize(new_size, resample=Resampling.BICUBIC)
    # paste to match left bottom
    position = (
        0,
        background_image.height - trajectory_image.height,
    )
    background_image.paste(trajectory_image, position, trajectory_image)
    background_image.save(output_image_path)


def main():
    this_file_directory = os.path.dirname(os.path.realpath(__file__))
    input_images = os.path.join(this_file_directory, "trajectory_images")
    output_images = os.path.join(
        this_file_directory, "trajectory_images_with_backgrounds"
    )
    if not os.path.exists(output_images):
        os.makedirs(output_images, exist_ok=True)

    for trajectory_image in os.listdir(input_images):
        trajectory_image_path = os.path.join(input_images, trajectory_image)
        if "room" in trajectory_image:
            background_image = "room_bg.png"
            ratio_hw = (0.9343853821, 0.9564983888)
        elif "cross" in trajectory_image:
            background_image = "cross_bg.png"
            ratio_hw = (0.94, 0.9398496241)
        else:
            raise ValueError(f"Invalid trajectory image {trajectory_image}")

        background_image_path = os.path.join(this_file_directory, background_image)
        output_image_path = os.path.join(
            output_images, f"{trajectory_image.split('.')[0]}_wbg.png"
        )
        print(
            f"Overlaying {trajectory_image_path} on {background_image_path} to {output_image_path}"
        )
        overlay_images(
            trajectory_image_path, background_image_path, output_image_path, ratio_hw
        )


if __name__ == "__main__":
    main()

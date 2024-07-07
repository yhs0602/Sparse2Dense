import os

from PIL import Image


def main():
    directory = os.path.dirname(os.path.realpath(__file__))
    trajectory_images = os.path.join(directory, "trajectory_images")
    final_image = None
    for trajectory_image in os.listdir(trajectory_images):
        if not "room" in trajectory_image:
            continue
        trajectory_image_path = os.path.join(trajectory_images, trajectory_image)
        if not final_image:
            final_image = Image.open(trajectory_image_path)
        else:
            img = Image.open(trajectory_image_path)
            final_image = Image.alpha_composite(final_image, img)
    final_image.save(os.path.join(directory, "all_trajectories.png"))


if __name__ == "__main__":
    main()

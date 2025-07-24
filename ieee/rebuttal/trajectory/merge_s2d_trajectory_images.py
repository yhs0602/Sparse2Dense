import os
from PIL import Image, ImageDraw, ImageFont

this_dir = os.path.dirname(os.path.abspath(__file__))


def merge_s2d_trajectory_images():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    s2d_dir = os.path.join(this_dir, "trajectory_images-s2d")

    try:
        font = ImageFont.truetype("arial.ttf", size=64)
    except:
        font = ImageFont.load_default(size=64)

    timing_val_to_images = {}
    for file in os.listdir(s2d_dir):
        if not file.endswith(".png"):
            continue
        print(file)
        s2d, transition_timing, timing, timing_val = file.split(".")[0].split("_")
        transition_timing = int(transition_timing)
        timing_val = int(timing_val)
        print(f"{s2d=} {transition_timing=} {timing=} {timing_val=}")
        if timing_val not in timing_val_to_images:
            timing_val_to_images[timing_val] = {}
        timing_val_to_images[timing_val][transition_timing] = file

    print(timing_val_to_images)
    output_dir = os.path.join(this_dir, "merged_trajectory_images-s2d")
    os.makedirs(output_dir, exist_ok=True)

    for timing_val, transition_timing_to_file in timing_val_to_images.items():
        print(f"{timing_val=} {transition_timing_to_file=}")
        canvas = Image.new("RGBA", (5000, 1000), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for transition_timing, file in transition_timing_to_file.items():
            print(f"{transition_timing=} {file=}")
            image = Image.open(os.path.join(s2d_dir, file))
            print(image.size)
            canvas.paste(image, ((transition_timing - 1) * 1000, 0))
            draw.text(
                ((transition_timing - 1) * 1000 + 50, 50),
                f"{transition_timing}",
                fill="black",
                font=font,
            )
        canvas.save(os.path.join(output_dir, f"merged_{timing_val}.png"))


if __name__ == "__main__":
    merge_s2d_trajectory_images()

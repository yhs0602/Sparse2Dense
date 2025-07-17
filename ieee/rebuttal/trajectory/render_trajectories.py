# Given trajectories, render them
# Group by algorithm
# From sampled_trajectories, from_to_seed_steps.csv.gz
# For each (from, to, steps)[6:seeds] -> render


import json
import math
import os

from PIL import Image
import pandas as pd


def aggregate_trajectories(trajectories_dir: str):
    pass


def real_steps(from_algo, to_algo, steps):
    if not from_algo:
        if steps in [0, 1000000, 2000000, 3000000]:
            return steps
        return None
    elif from_algo == "sparse":
        pass
    elif from_algo == "dense":
        pass
    else:
        raise ValueError(f"Invalid from_algo: {from_algo}")


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    trajectories_dir = os.path.join(current_path, "sampled_trajectories")

    data = {}
    for file in os.listdir(trajectories_dir):
        if not file.endswith(".csv.gz"):
            continue
        from_algo, to_algo, seed, steps = file.split("_")
        steps = int(steps.split(".")[0])
        if (from_algo, to_algo, steps) not in data:
            data[(from_algo, to_algo, steps)] = []

        data[(from_algo, to_algo, steps)].append(file)
    print(data)

    output_dir = os.path.join(current_path, "trajectory_images")
    os.makedirs(output_dir, exist_ok=True)

    sparse_dir = os.path.join(output_dir, "sparse")
    dense_dir = os.path.join(output_dir, "dense")
    s2d_dir = os.path.join(output_dir, "s2d")
    d2s_dir = os.path.join(output_dir, "d2s")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)
    os.makedirs(s2d_dir, exist_ok=True)
    os.makedirs(d2s_dir, exist_ok=True)

    render_pixels = True
    radius = 3

    for key, files in data.items():
        from_algo, to_algo, steps = key

        # Skip;
        if not from_algo:
            if steps > 3000000:
                print(f"Skipping {steps} > 3000000")
                continue
        else:
            if steps > 7000000:
                print(f"Skipping {steps + 3000000} > 7000000")
                continue

        image = Image.new("RGBA", (700, 1000), color=(255, 255, 255, 255))
        pixels = image.load()

        aggregations = {}

        alpha_increment = 1
        start_positions = []
        end_positions = []
        for file in files:
            full_file_path = os.path.join(trajectories_dir, file)
            df = pd.read_csv(full_file_path)
            # For each row, get the episode/positions and json.loads
            for _, row in df.iterrows():
                positions = json.loads(row["episode/positions"])
                if not positions:
                    continue
                start_positions.append(positions[0])
                end_positions.append(positions[-1])
                for position in positions:
                    # print(position)
                    x, y, z, yaw = position
                    x = int(x * 50)
                    z = int(z * 50)
                    aggregations[(x, z)] = aggregations.get((x, z), 0) + 1

        max_count = max(aggregations.values())
        for (x, z), count in aggregations.items():
            # alpha = int(min(count / max_count * 1000, 255))
            scaled = math.log1p(count) / math.log1p(max_count)
            scaled = scaled**0.5
            alpha = int(scaled * 255)
            pixels[x, z] = (0, 0, 0, alpha)

        for start_position in start_positions:
            x, y, z, yaw = start_position
            x = int(x * 50)
            z = int(z * 50)
            # draw a rectangle with radius
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx * dx + dz * dz <= radius * radius:
                        nx, nz = x + dx, z + dz
                        if nx < 0 or nx >= 700 or nz < 0 or nz >= 1000:
                            continue
                        pixels[nx, nz] = (255, 0, 0, 255)

        for end_position in end_positions:
            x, y, z, yaw = end_position
            x = int(x * 50)
            z = int(z * 50)
            # draw a rectangle with radius
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx * dx + dz * dz <= radius * radius:
                        nx, nz = x + dx, z + dz
                        if nx < 0 or nx >= 700 or nz < 0 or nz >= 1000:
                            continue
                        pixels[nx, nz] = (0, 0, 255, 255)

        # image.save(os.path.join(output_dir, f"{from_algo}_{to_algo}_{steps}.png"))

        # if render_pixels:
        #     r, g, b, a = pixels[x, z]
        #     new_alpha = min(a + alpha_increment, 255)
        #     pixels[x, z] = (0, 0, 0, new_alpha)
        # else:
        #     for dx in range(-radius, radius + 1):
        #         for dz in range(-radius, radius + 1):
        #             if dx * dx + dz * dz <= radius * radius:
        #                 nx, nz = x + dx, z + dz
        #                 r, g, b, a = pixels[nx, nz]
        #                 new_alpha = min(a + alpha_increment, 255)
        #                 pixels[nx, nz] = (0, 0, 0, new_alpha)
        # image.putpixel((x, z), (0, 0, 0, 100))
        # if from_algo is None, then the target directory is {to_algo}_sparse and {to_algo}_dense
        if not from_algo:
            if steps <= 3000000:
                if to_algo == "sparse":
                    image.save(os.path.join(sparse_dir, f"{steps}.png"))
                    image.save(os.path.join(s2d_dir, f"{steps}.png"))
                elif to_algo == "dense":
                    image.save(os.path.join(dense_dir, f"{steps}.png"))
                    image.save(os.path.join(d2s_dir, f"{steps}.png"))
                else:
                    raise ValueError(f"Invalid to_algo: {to_algo}")
            else:
                print(f"Skipping {steps} > 3000000")
        else:
            real_steps = steps + 3000000
            if real_steps > 10000000:
                print(f"Skipping {real_steps} > 10000000")
                continue
            if from_algo == "sparse":
                if to_algo == "sparse":
                    image.save(os.path.join(sparse_dir, f"{real_steps}.png"))
                elif to_algo == "dense":
                    image.save(os.path.join(s2d_dir, f"{real_steps}.png"))
                else:
                    raise ValueError(f"Invalid to_algo: {to_algo}")
            elif from_algo == "dense":
                if to_algo == "dense":
                    image.save(os.path.join(dense_dir, f"{real_steps}.png"))
                elif to_algo == "sparse":
                    image.save(os.path.join(d2s_dir, f"{real_steps}.png"))
                else:
                    raise ValueError(f"Invalid to_algo: {to_algo}")
            else:
                raise ValueError(f"Invalid from_algo: {from_algo}")

    # # Normalize the steps;
    # # from_algo: None, Sparse, Dense
    # # If the from_algo is None, then the steps 0, 1000000, 2000000, 3000000 is valid.
    # # If the from_algo is Sparse, then the steps
    # # to_algo: Sparse, Dense.

    # sparse_sparse_data = []
    # sparse_dense_data = []
    # dense_sparse_data = []
    # dense_dense_data = []

    # for key, files in data.items():
    #     for file in files:
    #         full_file_path = os.path.join(trajectories_dir, file)

    # # full_file_path = os.path.join(trajectories_dir, file)
    # # df = pd.read_csv(full_file_path)
    # # print(df.head())


if __name__ == "__main__":
    main()

import json

import pandas as pd


def main():
    # Read all_length.csv and select one row. Then save it to the output directory as json
    all_length = "all_lengths.csv"
    output_file = "selected_trajectory.json"
    data = pd.read_csv(all_length)
    first_row = data.iloc[0]
    positions = json.loads(first_row["episode/positions"])
    # Get action from positions.
    # Action = TURN_LEFT_90, TURN_RIGHT_90, MOVE_FORWARD
    # positions: [(x, y, z, yaw), ...]
    # Use yaw and position to get action
    actions = []
    for i in range(1, len(positions)):
        x, y, z, yaw = positions[i]
        prev_x, prev_y, prev_z, prev_yaw = positions[i - 1]
        if yaw == prev_yaw:
            actions.append("MOVE_FORWARD")
        elif yaw == prev_yaw + 90:
            actions.append("TURN_RIGHT_90")
        elif yaw == prev_yaw - 90:
            actions.append("TURN_LEFT_90")
        else:
            print(f"Unknown action: {yaw} {prev_yaw}")

    # Save the selected trajectory to the output directory
    selected_trajectory = {
        "positions": positions,
        "actions": actions,
    }
    with open(output_file, "w") as f:
        json.dump(selected_trajectory, f)


if __name__ == "__main__":
    main()

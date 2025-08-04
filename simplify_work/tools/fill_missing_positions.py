import json
from pathlib import Path
from tqdm import tqdm
import re
import numpy as np
from collections import defaultdict

INPUT_PATH = "training_dataset/0727pickplace/first100/meta/modified_tasks_truncated.jsonl"
OUTPUT_PATH = "training_dataset/0727pickplace/first100/meta/modified_tasks_filled.jsonl"
NEIGHBOR_RANGE = 3  # 前后各取 3 帧

def parse_position(text, key):
    match = re.search(rf"{key} at \(([-\d.,\s]+)\)m", text)
    if match:
        return tuple(map(float, match.group(1).split(",")))
    return None

def format_position(name, pos):
    return f"{name} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m"

def fill_missing(episode_data):
    filled_data = []
    n = len(episode_data)

    # 提取 gripper 和 object 位置
    gripper_positions = [parse_position(d['task'], "gripper") for d in episode_data]
    object_positions = [parse_position(d['task'], "the Pyramid-Shaped Sachet") for d in episode_data]

    for i in range(n):
        g_pos = gripper_positions[i]
        o_pos = object_positions[i]
        new_task = episode_data[i]["task"]

        def avg_from_neighbors(pos_list):
            neighbors = []
            for offset in range(1, NEIGHBOR_RANGE + 1):
                for j in [i - offset, i + offset]:
                    if 0 <= j < n and pos_list[j] is not None:
                        neighbors.append(pos_list[j])
                if len(neighbors) >= 2:
                    break
            if neighbors:
                return tuple(np.mean(neighbors, axis=0))
            return None

        # 如果缺失 gripper
        if g_pos is None:
            g_pos = avg_from_neighbors(gripper_positions)
            if g_pos:
                new_task += " " + format_position("gripper", g_pos)
        # 如果缺失 object
        if o_pos is None:
            o_pos = avg_from_neighbors(object_positions)
            if o_pos:
                if "gripper at" in new_task:
                    new_task += ", " + format_position("the Pyramid-Shaped Sachet", o_pos)
                else:
                    new_task += " " + format_position("the Pyramid-Shaped Sachet", o_pos)

        filled_data.append({
            "episode_index": episode_data[i]["episode_index"],
            "frame_index": episode_data[i]["frame_index"],
            "task": new_task.strip()
        })

    return filled_data

def main():
    path = Path(INPUT_PATH)
    with path.open("r") as f:
        lines = [json.loads(line) for line in f]

    episodes = defaultdict(list)
    for entry in lines:
        episodes[entry["episode_index"]].append(entry)

    output_lines = []
    for episode_index in tqdm(sorted(episodes), desc="Filling positions"):
        episode_data = sorted(episodes[episode_index], key=lambda x: x["frame_index"])
        filled = fill_missing(episode_data)
        output_lines.extend(filled)

    with open(OUTPUT_PATH, "w") as f:
        for entry in output_lines:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()

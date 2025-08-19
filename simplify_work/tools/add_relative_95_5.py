import json
prefix="training_dataset/0803_with_red/pickplace/first100/meta/"
old_file = prefix+"modified_states.jsonl"              # 原始文件路径
new_file = prefix+"mtask_relative_95.jsonl"   # 输出文件路径


with open(old_file, "r", encoding="utf-8") as fin, open(new_file, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        data = json.loads(line)
        episode_index = data["episode_index"]
        frame_index = data["frame_index"]
        state = data["state"]

        if state[-1] == 0.0:
            # 最后一位为0.0
            new_entry = {
                "episode_index": episode_index,
                "frame_index": 0,   # 按你的例子固定为0
                "task": "Pick up the pyramid-shaped sachet and place it into the box."
            }
        elif state[-1] == 1.0:
            # 最后一位为1.0
            x, y, z = state[6:9]  # 取倒数第4~2位（对应7-10位）
            new_entry = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task": (
                    "Pick up the pyramid-shaped sachet and place it into the box,"
                    f"sachet position relative to gripper is ({x:.3f}, {y:.3f}, {z:.3f})"
                )
            }
        else:
            raise ValueError("怎么有不是0的")
            continue  # 既不是0.0也不是1.0就跳过

        fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

print(f"处理完成，结果已保存到 {new_file}")

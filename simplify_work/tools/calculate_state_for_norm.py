import json
import numpy as np

# 函数1: 计算 state 最后一位是 1 的比例
def compute_last_one_ratio(filepath):
    total = 0
    count = 0
    with open(filepath, "r") as f:
        for line in f:
            record = json.loads(line)
            state = record["state"]
            total += 1
            if state[-1] == 1:
                count += 1
    return count / total if total > 0 else 0

# 函数2: 计算 state 后四位的 min, max, std, mean
def compute_last4_stats(adding_mode = "" ):
    prefix="training_dataset/0803_with_red/pickplace/first100/meta/"
    if adding_mode=="pure":
        filepath=prefix+"modified_states.jsonl"
    elif adding_mode=="grid_5cm":
        filepath=prefix+"modified_states_5cm.jsonl"
    else:
        raise AssertionError("adding_mode的模式有问题",adding_mode)

    values = []
    with open(filepath, "r") as f:
        for line in f:
            record = json.loads(line)
            state = record["state"]
            values.append(state[-4:])  # 取后四位
    arr = np.array(values)
    stats = {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }
    return stats

# # 使用示例
# if __name__ == "__main__":
#     filepath = "training_dataset/0803_with_red/pickplace/first100/meta/modified_states.jsonl"
#     ratio = compute_last_one_ratio(filepath)
#     stats = compute_last4_stats(filepath)
#     print("最后一位为1的比例:", ratio)
#     print("后四位统计:", stats)
def round_5cm_state(filepath = "training_dataset/0803_with_red/pickplace/first100/meta/modified_states.jsonl"):
    import json
    prefix="training_dataset/0803_with_red/pickplace/first100/meta/"
    input_file = prefix+"modified_states.jsonl"              # 原始文件路径
    output_file = prefix+"modified_states_5cm.jsonl"   # 输出文件路径

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            data = json.loads(line)  # 逐行读取 JSON
            state = data["state"]

            # 修改第 7-9 维度 (索引 6,7,8)
            for i in [6, 7, 8]:
                state[i] = round(state[i] / 0.05)

            # 写回新的 JSON 行
            fout.write(json.dumps(data) + "\n")
    print(f"修改完成，结果保存在 {output_file}")

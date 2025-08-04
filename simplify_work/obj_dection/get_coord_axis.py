import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def compute_axes(points_3d):
    """
    输入7个3D点，输出 origin, x_axis, y_axis, z_axis
    """

    # 提取关键点
    o  = np.array(points_3d[0])

    x1 = np.array(points_3d[1])
    x2 = np.array(points_3d[2])
    x3 = np.array(points_3d[3])

    y1 = np.array(points_3d[4])
    y2 = np.array(points_3d[5])
    y3 = np.array(points_3d[6])

    # 平均两个方向向量
    x_vec1 = x2 - x1
    x_vec2 = x3 - x2
    x_axis = normalize((x_vec1 + x_vec2) / 2)

    y_vec1 = y2 - y1
    y_vec2 = y3 - y2
    y_axis = normalize((y_vec1 + y_vec2) / 2)

    # 求z轴，并重新正交化y轴
    z_axis = normalize(np.cross(x_axis, y_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))  # 保证正交性

    # 输出格式化
    def format_array(name, arr):
        return f"{name} = np.array([{arr[0]: .8f}, {arr[1]: .8f}, {arr[2]: .8f}])"

    print(format_array("origin", o))
    print()
    print(format_array("x_axis", x_axis))
    print(format_array("y_axis", y_axis))
    print(format_array("z_axis", z_axis))


# 示例数据（你之前提供的7个点）
points = [
    (0.2416309171821922, -0.08227618725504726, 0.5907565169036388),  # o
    (0.2837373297661543, 0.022586133563891053, 0.5330665269866586),   # x1
    (0.2722652996890247, 0.0009972887983167311, 0.5464797513559461),  # x2
    (0.263313848990947, -0.020827370521146804, 0.5614070501178503),   # x3
    (0.24750629905611277, 0.07294383982662112, 0.4935711775906384),   # y1
    (0.2127215217333287, 0.08245761040598154, 0.48364155227318406),   # y2
    (0.18010479141958058, 0.08891348575707525, 0.47620831429958344)   # y3
]

compute_axes(points)

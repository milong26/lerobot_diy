import numpy as np
import cv2

def count_distance(image_tensor, depth_tensor):
    """
    输入：
        image_tensor: torch.Tensor，形状[C, H, W]，0~1范围（side图）
        depth_tensor: torch.Tensor，形状[C, H, W]，0~1范围（depth图，R=高8位，G=低8位）
    
    输出：
        返回红色与黄色目标中心点的欧几里得距离（单位米），如检测失败返回 None。
        同时保存检测图像为 get_red_and_yellow.png，并打印深度图RGB通道均值。
    """

    # Step 1: 图像还原
    color_img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    color_img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2HSV)

    # Step 2: 深度图还原（uint16，单位毫米）
    depth_img = (depth_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
    R, G, _ = depth_img[:, :, 0], depth_img[:, :, 1], depth_img[:, :, 2]
    depth_mm = ((R.astype(np.uint16) << 8) + G.astype(np.uint16))  # [H, W]
    depth_m = depth_mm.astype(np.float32) / 1000.0  # 转为米

    # Step 3: 输出RGB平均值
    r_mean = np.mean(R)
    g_mean = np.mean(G)
    b_mean = np.mean(depth_img[:, :, 2])
    print(f"Depth Image Channel Means - R: {r_mean:.2f}, G: {g_mean:.2f}, B: {b_mean:.2f}")

    # Step 4: 红色掩膜
    red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Step 5: 黄色掩膜（包含淡黄）
    yellow_mask = cv2.inRange(hsv, np.array([15, 30, 30]), np.array([40, 255, 255]))

    def find_center(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 100:
            return None, None
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        return (cx, cy), (x, y, x + w, y + h)

    red_center, red_box = find_center(red_mask)
    yellow_center, yellow_box = find_center(yellow_mask)

    if red_box:
        cv2.rectangle(color_img_bgr, (red_box[0], red_box[1]), (red_box[2], red_box[3]), (0, 0, 255), 2)
        cv2.circle(color_img_bgr, red_center, 5, (255, 255, 255), -1)
    if yellow_box:
        cv2.rectangle(color_img_bgr, (yellow_box[0], yellow_box[1]), (yellow_box[2], yellow_box[3]), (0, 255, 255), 2)
        cv2.circle(color_img_bgr, yellow_center, 5, (255, 255, 255), -1)

    # 保存检测图
    cv2.imwrite("get_red_and_yellow.png", color_img_bgr)

    # Step 6: 计算距离
    if red_center is None or yellow_center is None:
        return None

    red_z = depth_m[red_center[1], red_center[0]]
    yellow_z = depth_m[yellow_center[1], yellow_center[0]]

    if red_z == 0 or yellow_z == 0:
        return None

    red_pt = np.array([red_center[0], red_center[1], red_z])
    yellow_pt = np.array([yellow_center[0], yellow_center[1], yellow_z])
    dist = np.linalg.norm(red_pt - yellow_pt)

    return float(dist)

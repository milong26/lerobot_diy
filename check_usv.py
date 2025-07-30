import cv2

# 读取你保存的 HSV 图像
hsv_image = cv2.imread('debug_hsv.png')

# 回调函数：打印点击点的 HSV 值
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = hsv_image[y, x]
        print(f"HSV at ({x},{y}) = {tuple(hsv)}")

# 创建窗口并绑定鼠标事件
cv2.namedWindow("HSV Image")
cv2.setMouseCallback("HSV Image", on_mouse_click)

while True:
    cv2.imshow("HSV Image", hsv_image)
    if cv2.waitKey(1) & 0xFF == 27:  # 按下 Esc 退出
        break

cv2.destroyAllWindows()

import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"你点击的像素坐标是：({x}, {y})")

def main():
    image_path = "outputs/biaoding_imahge/sample_1.png"  # 换成你的图片路径
    img = cv2.imread(image_path)

    if img is None:
        print("无法加载图片，请检查路径是否正确。")
        return

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    print("点击图片任意位置以获取像素坐标。按任意键退出。")

    while True:
        cv2.imshow("Image", img)
        if cv2.waitKey(1) != -1:  # 按任意键退出
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

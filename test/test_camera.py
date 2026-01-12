# test_camera_params.py
import cv2
import time


def test_with_params(camera_index, width, height, fps):
    cap = cv2.VideoCapture(camera_index)

    # 设置参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 尝试MJPG格式
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    print(f"测试摄像头 {camera_index} - {width}x{height} @ {fps}fps")

    if cap.isOpened():
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                print(f"  成功！图像尺寸: {frame.shape}")
                cv2.imwrite(f"success_{camera_index}_{width}x{height}.jpg", frame)
                cap.release()
                return True
            time.sleep(0.1)
        print("  能打开但无法读取图像")
    else:
        print("  无法打开摄像头")

    cap.release()
    return False


# 测试不同配置
configs = [
    (0, 640, 480, 30),
    (0, 1280, 720, 30),
    (1, 640, 480, 30),
    (1, 1280, 720, 30),
]

for config in configs:
    if test_with_params(*config):
        break
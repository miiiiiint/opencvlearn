import cv2 as cv
import numpy as np

# 开启摄像头
cap = cv.VideoCapture(0)

while True:
    # 读取每一帧
    ret, frame = cap.read()
    
    # 如果正确读取帧，ret为True
    if not ret:
        print("无法读取摄像头画面")
        break

    # 将BGR图像转换为HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # 定义HSV中蓝色的范围
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # 对HSV图像进行阈值操作，只获取蓝色
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    
    # 对原图像和掩码进行位运算
    res = cv.bitwise_and(frame, frame, mask=mask)
    #当 cv.bitwise_and 被调用时，它会逐像素地对 src1 和 src2 进行按位与操作。如果提供了 mask，则只有那些掩码中对应像素值非零的位置才会执行按位与操作，其他位置的像素值在输出图像中保持不变或设置为0，取决于是否指定了输出图像。
    
    # 显示结果
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    
    # 按下'esc'键退出循环
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

# 关闭窗口
cv.destroyAllWindows()
# 释放摄像头资源
cap.release()

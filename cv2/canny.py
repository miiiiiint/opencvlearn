import cv2 as cv
import numpy as np

# 回调函数，但在这个例子中我们不需要它做任何事情
def nothing(x):
    pass

# 首先，读取图像
img = cv.imread('mp.png', 0)
if img is None:
    print("Error: 图像未成功加载，请检查文件路径。")
    exit()

# 创建一个名为'image'的窗口
cv.namedWindow('image')

# 创建两个轨迹条，分别用于Canny边缘检测的minval和maxval
cv.createTrackbar('maxval', 'image', 100, 1000, nothing)
cv.createTrackbar('minval', 'image', 200, 1000, nothing)

while(True):
    # 获取轨迹条的当前位置作为Canny算法的参数
    maxval = cv.getTrackbarPos('maxval', 'image')
    minval = cv.getTrackbarPos('minval', 'image')

    # 使用Canny算法进行边缘检测
    edges = cv.Canny(img, minval, maxval)

    # 将原始图像和边缘图像水平堆叠起来
    combined = np.hstack((img, edges))

    # 显示堆叠后的图像
    cv.imshow('image', combined)

    # 按下'ESC'键退出循环
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

# 销毁所有OpenCV窗口
cv.destroyAllWindows()

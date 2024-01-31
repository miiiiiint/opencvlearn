import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
while(True):
    # 一帧一帧捕捉
    ret, frame = cap.read()
    # 我们对帧的操作在这里
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 显示返回的每帧
    cv.imshow('frame',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# 当所有事完成，释放 VideoCapture 对象
cap.release()
cv.destroyAllWindows()
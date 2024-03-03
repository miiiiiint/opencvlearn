import numpy as np
import cv2 as cv

img = cv.imread('a.jpg',0)
cv.imshow('image',img)
k = cv.waitKey(0)& 0xFF
if k == 27: # ESC 退出
    cv.destroyAllWindows()
elif k == ord('s'): # 's' 保存退出
    cv.imwrite('messigray.png',img)
    cv.destroyAllWindows()
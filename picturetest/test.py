import cv2
import numpy
img = cv2.imread('c.jpg')
'''
if img is None:
    print("图像未加载")
else:
    cv2.imshow('img',img)
    '''
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2 as cv
img = cv.imread('b.jpg',0)
if img is None:
    print("图像未加载")
else:
    cv.imshow('img',img)
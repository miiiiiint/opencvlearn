import cv2 as cv

img = cv.imread('four.png',0)
retval, dst1 = cv.threshold(img, 125, 255, cv.THRESH_BINARY)
dst2 = cv.inRange(img,172,203)


cv.imshow('image1',dst1)
cv.imshow('image2',dst2)
cv.waitKey(0)
cv.destroyAllWindows()
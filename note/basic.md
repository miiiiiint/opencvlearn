>#  opencv learn notes
>#  opencv learn notes
> 显而易见，这是一个关于opencv学习的笔记

# GUI特性
## cv.imread()，cv.imshow()，cv.imwrite()
**imread**可以用于读取图像
`img = cv2.imread('messi5.jpg'，0)`
后面的参数可以填1，0，-1
分别代表加载彩色，灰度和包括alpha通道

> 第一个参数填图像的路径，当和程序文件在同一个目录下时可以直接填图像文件名，路径错误时不会报错，而是返回none

**imshow**可以用于显示图像
`    cv2.imshow('image'，img）`
image是窗口的名字，后面是一个变量，储存了jpg这个tupian

**imread**可以用于保存图片
`cv2.imwrite('46.png',img)`
这个操作将会把img写入到46.png中，如果没有该文件则会创建新文件，如果是是视频格式将会在视频末尾一帧一帧写入（这里不做讨论）
## 视频操作，cv2.VideoCapture() 
**cv2.VideoCapture() **可以用摄像头捕获视频，参数为0为默认计算机默认摄像头，1可以更换来源；
`cap = cv2.VideoCapture()`
当代码报错时，可以使用cap.isOpened()来检查是否成功初始化了，返回值是True，就没有问题，否则就要使用cap.open()初始化
可以使用cap.get(propId)来获取视频的一些参数信息，propId可以是0到18之间的任何数，每一个数代表一个属性,其中一些值可以使用cap.set(propId,value)来修改，例如cap.get(3)和cap.get(4)来查看每一帧的宽和高，默认是640x480
`cap.set(3,1080) cap.set(4,2000)`

读取文件中的视频，只需要把imread中的路径替换为视频路径，同时cv2.waitkey()会决定视频播放的速度，一般使用cv2.waitkey(25),这是一个恰好合适的速度（实际是指每一帧播放间隔时间为25ms）

保存视频我们可以**cv.imwrite()**来进行，这是一个比较困难的操作，即我们将创建一个新的VideoWriter 对象，然后将原视频的每一帧经过处理后（也可以不处理），写入进去。
下面是一段示例，大概的语法内容也在下面有所展现
```python
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
# 声明编码器和创建 VideoWrite 对象
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv.flip(frame,0)
        # 写入已经翻转好的帧
        out.write(frame)
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# 释放已经完成的工作
cap.release()
out.release()
cv.destroyAllWindows()

```

所有视频操作都需要在结束时使用`cap.release()`释放资源
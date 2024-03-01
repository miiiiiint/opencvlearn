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

#  shape
##### 使用shape函数会返回一个数组
前两项 shape[0] 和 shape[1] 表示图像的宽度和高度。
Shape[2] 代表通道数。
3 表示图像具有红绿蓝 (RGB) 3个通道。
```python
    shape[0]---宽
    shape[1]---高
    shape[2]---通道数`
```
## 按位运算
这里有一些关键字，如下
```python
bitwise_and
bitwise_or
bitwise_xor
bitwise_not
```
如名称所示来进行运算，分别是与，或，先与再非，非（自己的感觉不一定对）
> 然后后面的图像叠加有点不会，看一下视频先(完蛋，看了下视频觉得文档写得好垃圾，也可能是不是官方文档的原因，所以还是直接决定去看视频了（就是效率慢点）

下面是我的理解
视频上讲的的话，可以先分割（split）出3个通道b,g,r
然后b,g,r是数组（可能python基础还是差一些）
用类似于以下的方法进行更改
```python
b,g,r=cv2.split(img)
#这里是把bule，green两个通道的[0:100]范围的值（强度）改为255
b[0:100,0:100] = 255
g[0:100,0:100] = 255
#最后使用merge进行合成
img2 = cv2.merge((b,g,r))
```
> 我自己的理解，也不知道对不对
# 图像轮廓
emmm.我知道可以通过梯度算法(膨胀-腐蚀)来运算出图像的边缘
膨胀操作（我不会？
```python
cv2.dilate(img,kernel,iterations = n)
n = 1#n是一个常数决定腐蚀次数
kernel = np.ones((30,30),np.unit8)
#我也不知道这是什么，可能是我跳着看的原因？
```
腐蚀操作
>将dilate替换成erode即可

以此可以得到图像的边缘，但是还不算是轮廓，后面还有一个叫开运算和闭运算的东西
### 开运算
先腐蚀在膨胀
### 闭运算
现膨胀再腐蚀
## 礼帽与黑帽
有一说一我是看不懂这个的（后面还得再学反正，现在先过一遍再说
```python
礼帽 = 原始输入-开运算结果#突出了原图像更亮的区域
黑帽 = 闭运算-原始输入#突出了原图像更暗的区域
```

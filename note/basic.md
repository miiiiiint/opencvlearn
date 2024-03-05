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
cv2.VideoCapture()可以用摄像头捕获视频，参数为0为默认计算机默认摄像头，1可以更换来源；
`cap = cv2.VideoCapture()`
当代码报错时，可以使用cap.isOpened()来检查是否成功初始化了，返回值是True，就没有问题，否则就要使用cap.open()初始化
可以使用cap.get(propId)来获取视频的一些参数信息，propId可以是0到18之间的任何数，每一个数代表一个属性,其中一些值可以使用cap.set(propId,value)来修改，例如cap.get(3)和cap.get(4)来查看每一帧的宽和高，默认是640x480
`cap.set(3,1080) cap.set(4,2000)`

读取文件中的视频，只需要把imread中的路径替换为视频路径，同时cv2.waitkey()会决定视频播放的速度，一般使用cv2.waitkey(25),这是一个恰好合适的速度（实际是指每一帧播放间隔时间为25ms）

保存视频我们可以 **cv.imwrite()** 来进行，这是一个比较困难的操作，即我们将创建一个新的VideoWriter 对象，然后将原视频的每一帧经过处理后（也可以不处理），写入进去。
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

#### 嘶，剩下的gui操作，像划线，鼠标回调，还有调色板其实都写了的，结果没保存，全没了。害，有点痛苦面具

#核心操作
**本节的所有操作都和numpy有关，因为图片的本质就是3维数组，而numpy是一个很好用的数组处理工具。所以应该先对numpy的操作有一定了解**

## 访问和修改像素值
我们来看看原话
>首先，一张图像有自己的属性：宽，高，通道数。其中宽和高是我们肉眼可见的属性，而通道数则是图像能呈现色彩的属性。我们都知道，光学三原色是红色，绿色和蓝色，这三种颜色的混合可以形成任意的颜色。常见的图像的像素通道也是对应的R，G，B三个通道，在OpenCV中，每个通道的取值范围为0～255。(注：还有RGBA，YCrCb，HSV等其他图像通道表示方法)。即，一般彩色图像读进内存之后是一个h * w * c的矩阵，其中h为图像高(相当于矩阵的行)，w为图像宽(相当于矩阵列)，c为通道数。

好的接下来看一个例子
下面我们先加载一副彩色图像，更准确地说，是一副黄色图像，如图所示。
黄色为绿色和红色的混合。所以，该图像的所有像素值都应为R=255，G=255，B=0。

``` py
#这里是命令行输出
>>> import numpy as np
>>> import CV2
>>> img = CV2.imread("img/yellow.jpg")
>>> h,w,c = img.shape
#图像大小为128*128*3
>>> print(h,w,c)
128 128 3

```
从上面的代码中可以看到，您可以通过行和列坐标访问像素值。注意，对于常见的RGB 图像，OpenCV的**imread**函数返回的是一个蓝色(Blue)值、绿色(Green)值、红色(Red)值的数组，维度大小为3。而对于灰度图像，仅返回相应的灰度值。

>通过 **img.shape**，你可以获得关于图像维度的信息。对于彩色图像，shape 属性将返回一个包含三个元素的元组，分别是图像的高度（H）、宽度（W）和颜色通道数（C）。
高度（H）和宽度（W）表示图像的大小，即图像在垂直和水平方向上的像素数。
颜色通道数（C）通常是3，对应于RGB（红、绿、蓝）颜色空间的三个通道。有时，如果图像包含透明度信息，则可能会有第四个通道，即RGBA中的A代表透明度。

------------

下面来看另外一个例子
``` py
>>> img[100,100]
#OpenCV的读取顺序为B，G，R，由于图像所有像素为黄色，因此，G=255，R=255
array([  0, 255, 255], dtype=uint8)

# 仅访问蓝色通道的像素
>>> blue = img[100,100,0]
>>> print(blue)
0

```

这里解释一下仅访问蓝色通道的像素是怎么做到的
**仅访问特定颜色通道的像素值：**
当你想要获取同一个像素点的特定颜色通道的值时，可以通过指定该通道的索引来实现。**在 BGR 格式中，索引 0 代表蓝色通道，1 代表绿色通道，2 代表红色通道。**

所以，**img[100, 100, 0]** 获取的是第 100 行、第 100 列像素点的蓝色通道的值。在你的例子中，这个值是 0，意味着在该像素点上，没有蓝色成分。

------------

好的下面是一个新例子，介绍如何使用numpy对像素进行修改
```py
#访问 红色通道 的值
>>>img.item(10,10,2)
59

#修改 红色通道 的值
>>>img.itemset((10,10,2),100)
>>>img.item(10,10,2)
100
```
`img.itemset((10,10,2), 100)` 这行代码用于修改同一个像素点（第 10 行、第 10 列）红色通道的值。这里，itemset 方法接受两个参数：第一个参数是一个包含行、列和颜色通道索引的元组 (10, 10, 2)；第二个参数是你想要设置的新值，在这个例子中是 100。执行这个操作后，该像素点的红色通道值将被更改为 100。

## 访问图像属性
与一般的**numpy.array**一样，可以通过 **img.shape** 访问图像的形状。它返回一组由图像的行、列和通道组成的元组（如果图像是彩色的）：
```py
>>>print(img.shape)
(128,128,3)
```
通过 **img.size** 访问图像的总像素数：
```py
>>>print(img.size)
562248
```
图像数据类型可以由 **img.dtype** 获得：
```py
>>>print(img.dtype)
UINT8
```
>注意 img.dtype 在调试时非常重要，因为 OpenCV-Python 代码中的大量错误由无效的数据类型引起。
>>（目前我也不知道我也不知道会引起怎样的错误，所以先记着）

## 图像中的感兴趣区域
没说什么，大概意思就是可以通过赋值的方式来实现图像的覆盖。

## 拆分和合并图像通道
直接上例子吧
```py
>>>b,g,r = CV.spilt(img)
>>>img = CV.merge((b,g,r))
```
或者
`b = img[:,:,0]`
前面两个：指的是选定所有像素，后面的0指的是蓝色通道，这样b就是img的单独的蓝色通道了。
然后，**cv.merge**是合并通道的一个函数，不过通道一定要按照b,g,r的顺序。
>ps：CV.spilt()是一项代价高昂的操作（就时间而言）。所以只在你需要时再这样做，否则使用 Numpy 索引。
>>具体原因我也不知道，这可能和函数背后的实现原理有关，这里不做了解。

## 制作图像边界（填充）
**CV.copyMakeBorder()**
`cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)`

**src**：输入图像。
**top, bottom, left, right**：分别表示顶部、底部、左侧和右侧的边框宽度，单位是像素。
**borderType**：定义了边框的类型。OpenCV提供了几种不同的边框类型，包括：
>**cv2.BORDER_CONSTANT**：添加一个常数值颜色的边框，这种情况下需要额外指定 value 参数来设定边框的颜色。
**cv2.BORDER_REFLECT**：边框会以镜像的方式被复制，例如：fedcba|abcdefgh|hgfedcb。
**cv2.BORDER_REFLECT_101 或 cv2.BORDER_DEFAULT**：类似于 BORDER_REFLECT，但稍有不同，例如：gfedcb|abcdefgh|gfedcba。
**cv2.BORDER_REPLICATE**：复制最边缘的像素，例如：aaaaaa|abcdefgh|hhhhhhh。
**cv2.BORDER_WRAP**：不常用，它会以一种特殊方式包装图像，例如：cdefgh|abcdefgh|abcdefg。
**value**：当边框类型为 **cv2.BORDER_CONSTANT** 时，该参数用于设定边框的颜色。颜色格式应与图像的通道数相匹配（例如，对于三通道的彩色图像，颜色值可以是 [B, G, R]）。

下面是一个例子
```py
import cv2

# 加载图像
image = cv2.imread('path/to/your/image.jpg')

# 添加边框
blue = [255, 0, 0] # BGR
borderedImage = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=blue)

# 显示结果
cv2.imshow('Bordered Image', borderedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
------

## 图像的像素运算
**cv.add()，cv.addWeighted()**

您可以通过 OpenCV 函数，cv.add()或简单地通过 numpy 操作将两个图像相加，res = img1 + img2。两个图像应该具有相同的深度和类型，或者第二个图像可以是像素值，比如(255,255,255)，白色值。

需要注意的，我们一般使用cv.add()函数进行运算，因为直接使用numpy处理8位无符号整型时，在溢出时会进行模运算，以至于得到不想要的结果。

`cv.addWeighted(src1, alpha, src2, beta, gamma, dst)`
rc1：第一幅输入图像。
alpha：第一幅图像的权重。
src2：第二幅输入图像。
beta：第二幅图像的权重。
gamma：一个加到每个总和的标量值。
dst：输出的图像。

公式如下
`dst = src1 * alpha + src2 * beta + gamma`
就不搞例子了，这个还是很好理解的

------------


## 按位运算
（~~不懂原理是什么，不知道每个像素点背后的数组发生了怎么样的运算，所以算了算了，知道结果是什么就行了~~）
这里有一些关键字，如下
```python
bitwise_and
bitwise_or
bitwise_xor
bitwise_not
```
**按位与（bitwise_and）**：这个操作对两个图像（或图像与掩码）中的每个像素的二进制表示进行按位与操作。它可以用来提取图像的特定部分。例如，通过使用形状为特定图案的掩码，可以从原图中提取出该图案对应的区域。

**按位或（bitwise_or）**：此操作对两个图像中的每个像素的二进制表示进行按位或操作。这可以用于合并、添加或叠加图像的某些部分。

按位非（bitwise_not）：这个操作将图像中每个像素的二进制表示进行按位取反操作。它常用于图像颜色的反转，比如将黑色变为白色，白色变为黑色。

**按位异或（bitwise_xor）**：此操作对两个图像中的每个像素的二进制表示进行按位异或操作。这可以用于图像的某些特殊效果，比如找出两个图像不同的部分。
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
> 懂了，实际上就是意思上的那样，对每个值进行位操作，只需要知道255的二进制码是111111111而0的二进制码是00000000.

-----

## 性能测量和改进技术

~~有什么好说的。。。。我还没学完呢，优化执行速度什么的目前不予考虑好吧。~~
# 图像处理
**这基本上可以算的上最重要的一节了，所以会尽量搞懂每个点**

##改变颜色空间
**cv.cvtColor()，cv.inRange()**
在 OpenCV 中有超过 150 种颜色空间转换的方法。但我们仅需要研究两个最常使用的方法，他们是 BGR 到 Gray，BGR 到 HSV。

然后在cv2文件中写了一个bgr到hsv的例子。然后在hsv空间中提取出了图片的蓝色部分。。
另外，下面展示了怎么通过 **cv.cvtColor()** 来找到hsv空间的颜色值
```py
>>> green = np.uint8([[[0,255,0 ]]])
>>> hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
>>> print( hsv_green )
[[[ 60 255 255]]]
```

## 图像的几何变换
**cv.warpAffine** ， **cv.warpPerspective**， **cv.resize()** 
### 缩放
`cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
`
src：输入图像。
dsize：目标图像大小，以(width, height)的形式表示。如果设置为None，则必须通过fx和fy参数指定缩放比例。
dst：输出图像。这是一个可选参数。
fx和fy：分别沿x轴和y轴的缩放比例。如果dsize为None，则这两个参数必须被指定。否则，它们是可选的。
interpolation：插值方法。这是一个非常重要的参数，因为它决定了在缩放过程中如何计算新像素的值。OpenCV提供了多种插值方法，例如：
>cv2.INTER_LINEAR：线性插值（默认值），适用于大多数情况。
cv2.INTER_NEAREST：最近邻插值，速度最快，但质量最低，可能会导致锯齿边缘。
cv2.INTER_AREA：区域插值，适用于缩小图像时。
cv2.INTER_CUBIC：三次插值，比线性插值慢，但质量更好。
cv2.INTER_LANCZOS4：Lanczos插值，使用8x8邻域，提供高质量的结果。
### 平移，旋转，仿射和透视
这里似乎设计到数学中的矩阵知识。
![alt text](https://apachecn.github.io/opencv-doc-zh/docs/4.0.0/img/Geometric_Transformations_fomula_1.png)
所以我们先了解 **cv.warpAffine** 这个函数
`cv2.warpAffine(src, M, dsize, dst=None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
`
>src：输入图像。
**M：2x3的变换矩阵。**
dsize：输出图像的大小，以(width, height)形式表示。
dst：输出图像。这是一个可选参数。
flags：插值方法。与cv.resize()函数中的插值方法相同，常用的有cv2.INTER_LINEAR、cv2.INTER_NEAREST等。 **（因为在实际操作时，很有可能无法将像素点映射到对应的整数点，所以这个时候就需要差值处理）** 
borderMode：边界像素模式。这个参数定义了图像边界的处理方式，常见的有cv2.BORDER_CONSTANT（添加固定颜色边界）、cv2.BORDER_REFLECT、cv2.BORDER_WRAP等。
borderValue：当borderMode=cv2.BORDER_CONSTANT时使用的边界颜色值，默认为0，即黑色。

这个opencv通过这个M来计算每一个像素的落点位置，那么，我们该怎么真的这个M呢？
~~那当然是直接算啦~~
咳咳，其实opencv提供了一些函数来运算M，比如cv.getAffineTransform ， cv.getRotationMatrix2D， cv.getPerspectiveTransform。他们分别是计算旋转矩阵，仿射变化矩阵和透视变换的函数。不过透视变换要使用 **cv.warpPerspective** 而不是 **cv.warpAffine** 记着就好，具体参数就不在说明了，因为要用的时候也很好查到。

## 图像阈值
如果像素值大于阈值，则会被赋为一个值（可能为白色），否则会赋为另一个值（可能为黑色）。使用的函数是 **cv.threshold** 。
`retval, dst = cv2.threshold(src, thresh, maxval, type)
`
>src：输入图像，必须是单通道的灰度图。

>thresh：阈值，用于与像素值比较的数值。

>maxval：当像素值超过（有时是低于，取决于阈值类型）阈值时应该被赋予的最大值。

>type：阈值类型，决定了具体的阈值处理策略。OpenCV 提供了多种阈值类型：
>>cv2.THRESH_BINARY：如果像素值超过阈值，则被赋予 maxval，否则为 0。
cv2.THRESH_BINARY_INV：THRESH_BINARY 的反转，如果像素值超过阈值，则为 0，否则为 maxval。
cv2.THRESH_TRUNC：如果像素值超过阈值，则被设置为阈值，否则保持不变。
cv2.THRESH_TOZERO：如果像素值低于阈值，则设置为 0，否则保持不变。
cv2.THRESH_TOZERO_INV：THRESH_TOZERO 的反转，如果像素值超过阈值，则设置为 0，否则保持不变。

>retval：实际使用的阈值，这在自适应阈值处理（例如 Otsu's 方法）中特别有用。

>dst：输出图像，与输入图像有相同的大小和类型。
## 自适应阈值
**cv.adaptiveThreshold**
`dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
`
具体可以看写出的代码，这个函数会根据周围像素的平均值来生成阈值，更加适合有光照变化的场景。
```py
# 应用自适应阈值处理
thresholded_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)

# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('Adaptive Thresholded Image', thresholded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 平滑图像
opencv 提供了函数 **cv.filter2D()**，用于将内核与图像卷积起来。
操作如下：将该内核中心与一个像素对齐，然后将该内核下面的所有 25 个像素相加，取其平均值，并用新的平均值替换这个25x25窗口的中心像素。
>有一说一，并不是很懂原理，因为我还没有学过卷积。
### 图像模糊（图像平滑）
有下面几种
#### 均值模糊
**cv.blur()** 或 **cv.boxFilter()**
顾名思义，取像素点的均值。

#### 高斯模糊
嘶，不知道怎么处理的，不过用法如下
`blur = cv.GaussianBlur(img,(5,5),0)
`
#### 中值模糊
**cv.medianBlur()**
顾名思义，取周围像素点的中值。

#### 双边滤波
**cv.bilateralFilter()** 在保持边缘锐利的同时，对噪声去除非常有效。但与其他过滤器相比，操作速度较慢。
现在先只记特点，原理不知道思密达。

## 形态转换
### 腐蚀
我的理解：一个方格扫描图像中的像素，所有像素都为“1”才保留，否则就舍弃了。
结果是小了一圈。
### 膨胀
我的理解：一个方格扫描图像中的像素，只要一个像素为“1”，那这个方格里的所有像素都变成“1”。
结果是大了一圈。

### 开运算
先腐蚀后膨胀。正如我们上面所解释的，它对消除噪音很有用。在这里，`cv.morphologyEx(img, cv.MORPH_OPEN, kernel)`。
### 闭运算
关闭与打开相反，膨胀后腐蚀。它在填充前景对象内的小孔或对象上的小黑点时很有用
`cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)`
### 顶帽与黑帽
```python
顶帽 = 原始输入-开运算结果#突出了原图像更亮的区域
黑帽 = 闭运算-原始输入#突出了原图像更暗的区域
```
### 嘶，图像梯度，用了一大堆词语来试图让我理解这个东西，emmmm，但最后还是没懂，~~数学实在是太让人头大了~~
 OpenCv 提供三种类型的梯度滤波器或高通滤波器，Sobel、Scharr 和 Laplacian。（有一说一，还是不太懂，不过还是记下来先）
 ```py
 import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('box.png',0)
# Output dtype = cv.CV_8U
sobelx8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()
 ```

 在上一个示例中，输出数据类型是 cv.CV_8U或 np.uint8。但这有一个小问题。黑白过渡为正斜率（有正值），而白黑过渡为负斜率（有负值）。所以当你把数据转换成 np.uint8 时，所有的负斜率都变成零。简单来说，你失去了边缘。
 >这个就是不同数据类型的范围不同呗，np.uint8里面没有负值，所以所有的负值在转换时都会变为0.

 ## 边缘检测
 啊啊啊，终于到边缘检测了
 **cv.Canny（）**
 Canny 边缘检测是一种流行的边缘检测算法。它是由 John F. Canny 在 1986 年提出。
 首先第一步，我们需要对图像进行降噪，也就是使用我们前面所学到的高斯滤波器。因为边缘检测很容易受到图像中的噪声影响，所以这一步是很有必要的。

 然后，在水平和垂直方向上用 Sobel 内核对平滑后的图像进行滤波，这是为了寻找到图像的强度梯度，就像图像梯度里面所做的一样。

 嘶，再然后就有些头疼了。这一步是叫 **非最大抑制** ，意思就是在获得梯度幅度和方向之后，完成图像的全扫描以去除可能不构成边缘的任何不需要的像素。为此，在每个像素处，检查像素是否是其在梯度方向上的邻域中的局部最大值。（人话：不构成边缘的像素会被抑制，也就是设为0）

 最后是 **滞后阈值** ，简单来讲，就是通过算法，来去除掉不是边缘的边缘，这个算法需要两个阈值来决定去掉哪些边缘，分别是最大阈值maxval和最小阈值minval。下面看原文
 ![alt text](https://apachecn.github.io/opencv-doc-zh/docs/4.0.0/img/Image_canny_hysteresis.jpg)
 边缘 A 高于 maxVal，因此被视为“确定边缘”。虽然边 C 低于 maxVal，但它连接到边 A，因此也被视为有效边，我们得到完整的曲线。但是边缘 B 虽然高于 minVal 并且与边缘 C 的区域相同，但它没有连接到任何“可靠边缘”，因此被丢弃。因此，我们必须相应地选择 minVal 和 maxVal 才能获得正确的结果。

>#  opencv learn notes

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
# 按位运算
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



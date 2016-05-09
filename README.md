# SSDA实现
## 描述
SSDA由两部分网络组成：一个考虑稀疏权重的DA和一个SDA，前者使得SSDA和传统SDA有所不同。


两部分网络分别如下（由caffe提供的draw.py绘制）：
![网络一：DA](http://7xljx0.com1.z0.glb.clouddn.com/phase1.png?imageView/2/w/619/q/90)
[原图](http://7xljx0.com1.z0.glb.clouddn.com/phase1.png)
![网络二：三层SDA](http://7xljx0.com1.z0.glb.clouddn.com/SDA_train.png?imageView/2/w/619/q/90)
[原图](http://7xljx0.com1.z0.glb.clouddn.com/SDA_train.png)

## 预处理
训练集我是从网上找了50张图片，然后resize到了80*50大小作为无噪声的输入，同时也把这些图加上噪声作为另一个输入。
由于图片维度太高，所以将有噪声和无噪声的图片集都分成10 * 10大小的patch
最后把两个输入存到一个hdf5里
## 训练
### 第一个网络
第一个网络的loss function为：

![](http://7xljx0.com1.z0.glb.clouddn.com/Screen%20Shot%202016-05-09%20at%204.33.42%20PM.png?imageView/2/w/619/q/90)
由于caffe中没有KL divergence的loss layer,所以这层是我自己定义的，我参考了caffe样例里的loss.py，照猫画虎做了一个kllosslayer。

![](http://7xljx0.com1.z0.glb.clouddn.com/Screen%20Shot%202016-05-09%20at%204.38.42%20PM.png?imageView/2/w/619/q/90)

其他的layer由于有现成提供，所以就直接使用。
在定义好solver和layer后，就用caffe的命令行工具进行参数的训练。

### 中间结果的处理
由于第二部分网络需要第一部分的中间结果h(x)和h(y)，所以我使用了caffe的python接口，定义了部分网络后，载入训练好的模型，进行forward的操作得到所需要的结果。

### 第二个网络
第二部分的定义方法比较常规，就是按照它的结构写出相应的prototxt文件。
同样使用命令行工具进行训练

### 效果
训练好两个caffemodel后，我使用一个python脚本来完成使用caffemodel得到目标结果的过程
主要思路如下：
1. 将输入图片分成许多patch
2. 将patch输入网络1的前一部分，获得中间结果tmp1
3. 将tmp1输入网络2，获得中间结果tmp2
4. 将tmp2输入网络1的后半部分，获得中间结果tmp3
5. 将tmp3还原成一张原来大小的图片

### 遇到的问题和解决的办法
#### 如何得到训练的结果
最开始是照着参考官方样例在做，但是一直没有搞懂怎么获取训练的结果（因为样例使用的都是命令行工具，train和test是一起做的，不是很明白中间的过程）。
因为资料比较零碎不很好找，最后是在官方论坛上提问弄懂的。
#### 不收敛的情况
在训练DA的时候，我很快遇到了参数不收敛的情况。后来发现是我自己定义的kllosslayer的backpropate写错了。
#### 结果一片糊
把训练集的图片放进网络中forward都无法得出和原来相近的图片，感觉一片糊。不知道是正则项系数太大了还是kllayer出了问题，抑或是patch搞得太小了。

### 目前没做和不懂的问题
1. 没有调参，不是很清楚正则项和KL的参数应该怎么调(电脑很慢)
2. 没有使用L-BFGS algorithm回归
3. 没有用PSNR评测 
4.  不是很懂finetune是什么意思

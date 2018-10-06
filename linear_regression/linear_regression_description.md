# 线性回归
**线性回归可以通过升维或升幂方式拟合非线性可分数据。**
## 1. 理论基础：
* 大数定律：在试验不变的条件下，重复试验多次，随机事件的频率近似于它的概率。
* 中心极限定理：一些现象受到许多相互独立的随机因素的影响，如果每个因素所产生的影响都很微小时，总的影响可以看作是服从正态分布的。
* 线性函数：一阶（或更低阶）多项式，或零多项式。
* 线性回归模型：利用线性函数对一个或多个自变量（x）和因变量（y）之间的关系进行拟合的模型。
![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/linear_1.png?raw=true)
## 2. 定义：
<a href="https://www.codecogs.com/eqnedit.php?latex=y=w_{0}&plus;w_{1}*x_{1}&plus;w_{2}*x_{2}&plus;...&plus;w_{n}*x_{n}=W^{T}*X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=w_{0}&plus;w_{1}*x_{1}&plus;w_{2}*x_{2}&plus;...&plus;w_{n}*x_{n}=W^{T}*X" title="y=w_{0}+w_{1}*x_{1}+w_{2}*x_{2}+...+w_{n}*x_{n}=W^{T}*X" /></a>

其中，<a href="https://www.codecogs.com/eqnedit.php?latex=x_{1}...x_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{1}...x_{n}" title="x_{1}...x_{n}" /></a>表示样本的n维特征，<a href="https://www.codecogs.com/eqnedit.php?latex=w_{0},...w_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{0},...w_{n}" title="w_{0},...w_{n}" /></a>表示不同特征的权重。
  
### 线性回归模型： 
* 简单线性回归:数据集中样本只有一个特征，类似于一元线性函数；
* 多元线性回归:数据集中样本包含有多个特征；
### 广义线性回归模型：
* <a href="https://www.codecogs.com/eqnedit.php?latex=y=g^{-1}(W^{T}*X&plus;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=g^{-1}(W^{T}*X&plus;b)" title="y=g^{-1}(W^{T}*X+b)" /></a>
* 对数线性回归：<a href="https://www.codecogs.com/eqnedit.php?latex=lny=W^{T}*X&plus;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?lny=W^{T}*X&plus;b" title="lny=W^{T}*X+b" /></a>形式上仍是线性回归，实质上是在求取输入空间到输出空间的非线性映射。
* 倘若所做的是分类任务，则只需找一个单调可微函数将分类任务的真实标记y与线性回归模型的预测值联系起来。对于二分类任务，常使用对数几率函数（sigmoid函数），这也就是逻辑回归的来源。
## 3. 线性回归的目标函数
* 预测值y'和真实值y之间的差异，使用<a href="https://www.codecogs.com/eqnedit.php?latex=(y^{'}-y)^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(y^{'}-y)^{2}" title="(y^{'}-y)^{2}" /></a>来表示。
* 如何衡量这种差异，引入损失函数，即<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{80}&space;J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(y^{'(i)}-y^{(i)})^{2}=\frac{1}{2m}\sum_{i=1}^{m}(W^{^{T}}*X^{(i)}&plus;b-y^{(i)})^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(y^{'(i)}-y^{(i)})^{2}=\frac{1}{2m}\sum_{i=1}^{m}(W^{^{T}}*X^{(i)}&plus;b-y^{(i)})^{2}" title="J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(y^{'(i)}-y^{(i)})^{2}=\frac{1}{2m}\sum_{i=1}^{m}(W^{^{T}}*X^{(i)}+b-y^{(i)})^{2}" /></a>我们要做的，就是找到最好的W和b。

## 4. 参数的求解

### 最小二乘法与极大似然估计
* 上述目标函数是基于均方误差最小化来进行模型求解的方法，也被称为“最小二乘法”。线性回归模型就是使用最小二乘“参数估计”进行求解w,b。
而从概率的角度来看待这个问题，求w,b的极大似然估计，其结果可以在数学上等价于求w,b的最小二乘法。
### 梯度下降法
梯度下降法基于的思想为：要找到某函数的极小值，则沿着该函数的梯度方向寻找。若函数为凸函数且约束为凸集，则找到的极小值点则为最小值点。
梯度下降基本算法为： 首先用随机值填充θ（这被称为随机初始化），然后逐渐改进，每次步进一步(步长α)，每一步都试图降低代价函数，直到算法收敛到最小。








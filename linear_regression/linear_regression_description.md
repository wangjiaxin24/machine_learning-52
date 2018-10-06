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
  
### 线性回归又包括： 
* 简单线性回归:（数据集D中样本是有1个属性所描述，类似于一元线性函数 y=wx+by=wx+b ）；
* 多元线性回归:（数据集D中样本由d个属性所描述， y=WTX+by=WTX+b ）；
* 广义线性回归：如对数几率回归，
## 2. 最小二乘法与极大似然估计





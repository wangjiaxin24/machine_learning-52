# 线性回归
**线性回归可以通过升维或升幂方式拟合非线性可分数据。**
## 1. 理论基础：
* 大数定律：在试验不变的条件下，重复试验多次，随机事件的频率近似于它的概率。
* 中心极限定理：一些现象受到许多相互独立的随机因素的影响，如果每个因素所产生的影响都很微小时，总的影响可以看作是服从正态分布的。
* 线性函数：一阶（或更低阶）多项式，或零多项式。
* 线性回归模型：利用线性函数对一个或多个自变量（x）和因变量（y）之间的关系进行拟合的模型。
![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/linear_1.png?raw=true)

## 2. 模型分类：
<a href="https://www.codecogs.com/eqnedit.php?latex=y=w_{0}&plus;w_{1}*x_{1}&plus;w_{2}*x_{2}&plus;...&plus;w_{n}*x_{n}=W^{T}*X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=w_{0}&plus;w_{1}*x_{1}&plus;w_{2}*x_{2}&plus;...&plus;w_{n}*x_{n}=W^{T}*X" title="y=w_{0}+w_{1}*x_{1}+w_{2}*x_{2}+...+w_{n}*x_{n}=W^{T}*X" /></a>

其中，<a href="https://www.codecogs.com/eqnedit.php?latex=x_{1}...x_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{1}...x_{n}" title="x_{1}...x_{n}" /></a>表示样本的n维特征，<a href="https://www.codecogs.com/eqnedit.php?latex=w_{0},...w_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{0},...w_{n}" title="w_{0},...w_{n}" /></a>表示不同特征的权重。
  
### （1）线性回归模型： 
* 简单线性回归:数据集中样本只有一个特征，类似于一元线性函数；
* 多元线性回归:数据集中样本包含有多个特征。
### （2）广义线性回归模型：
* <a href="https://www.codecogs.com/eqnedit.php?latex=y=g^{-1}(W^{T}*X&plus;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=g^{-1}(W^{T}*X&plus;b)" title="y=g^{-1}(W^{T}*X+b)" /></a>
* 对数线性回归：<a href="https://www.codecogs.com/eqnedit.php?latex=lny=W^{T}*X&plus;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?lny=W^{T}*X&plus;b" title="lny=W^{T}*X+b" /></a>形式上仍是线性回归，实质上是在求取输入空间到输出空间的非线性映射。
* 倘若所做的是分类任务，则只需找一个单调可微函数将分类任务的真实标记y与线性回归模型的预测值联系起来。对于二分类任务，常使用对数几率函数（sigmoid函数），这也就是逻辑回归的来源。

## 3.损失函数
### （1）最小二乘法
* 预测值y'和真实值y之间的差异，使用<a href="https://www.codecogs.com/eqnedit.php?latex=(y^{'}-y)^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(y^{'}-y)^{2}" title="(y^{'}-y)^{2}" /></a>来表示。
![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/linear_2.png?raw=true)
* 为减少样本量对损失函数的影响，我们可以选择除以样本量m。

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(y^{'(i)}-y^{(i)})^{2}=\frac{1}{2m}\sum_{i=1}^{m}(W^{^{T}}*X^{(i)}&plus;b-y^{(i)})^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(y^{'(i)}-y^{(i)})^{2}=\frac{1}{2m}\sum_{i=1}^{m}(W^{^{T}}*X^{(i)}&plus;b-y^{(i)})^{2}" title="J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(y^{'(i)}-y^{(i)})^{2}=\frac{1}{2m}\sum_{i=1}^{m}(W^{^{T}}*X^{(i)}+b-y^{(i)})^{2}" /></a>
### （2）极大似然估计
![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/linear_3.png?raw=true)
![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/linear_4.png?raw=true)

## 4.损失函数中添加正则项
![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/linear_5.png?raw=true)
![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/linear_6.png?raw=true)
[为什么L1正则化可以产生稀疏模型（L1是怎么让系数等于零的），以及为什么L2正则化可以防止过拟合?](https://blog.csdn.net/jinping_shi/article/details/52433975)

## 5. 模型参数的求解
### （1）求导法
* 对各参数求偏导，并使其偏导为0，进而求出最优的参数。但此方法不适用于不可导函数，且计算量过大。
### （2）梯度下降法
* 将x增加一个维度x(n+1)=(,1),w也增加一个维度w(n+1)=（，b）于是原始的目标函数可以视为求解关于模型参数θ=w+w(n+1)。
梯度下降法基于的思想为：要找到某函数的极小值，则沿着该函数的梯度方向寻找。若函数为凸函数且约束为凸集，则找到的极小值点则为最小值点。
梯度下降基本算法为： 首先用随机值填充θ（这被称为随机初始化），然后逐渐改进，每次步进一步(步长α)，每一步都试图降低代价函数，直到算法收敛到最小。

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\theta&space;=\theta&space;&plus;\alpha&space;\bigtriangledown&space;_{\theta&space;}J(\theta&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;\theta&space;=\theta&space;&plus;\alpha&space;\bigtriangledown&space;_{\theta&space;}J(\theta&space;)" title="\theta =\theta +\alpha \bigtriangledown _{\theta }J(\theta )" /></a>
* 常见的梯度下降法：批量梯度下降(BGD)  随机梯度下降(SGD) 小批量梯度下降(MBGD)
* [参考爖的笔记](https://note.youdao.com/share/?id=981825c617d47c10f4e0c373e8b7bfff&type=note#/)

**线性回归模型的训练就是在确定损失函数的情况下，以特定的方式寻找最优参数w,b的过程（b是截距也称为bias）。**

## 6. sklearn.linear_model.LinearRegression中的参数
**linearRegression(fit_intercept=True,nomalize=False,copy_X=True,n_jobs=None)**
* （1）主要参数说明：

**fit_intercept**: 默认为True,是否计算该模型的截距。如果使用中心化的数据，可以考虑设置为False,不考虑截距。注意这里是考虑，一般还是要考虑截距

normalize: 默认为false. 当fit_intercept设置为false的时候，这个参数会被自动忽略。如果为True,回归器会标准化输入参数：减去平均值，并且除以相应的二范数。当然啦，在这里还是建议将标准化的工作放在训练模型之前。通过设置sklearn.preprocessing.StandardScaler来实现，而在此处设置为false

**copy_X** : 默认为True, 否则X会被改写

**n_jobs**: int 默认为1. 当-1时默认使用全部CPUs ??(这个参数有待尝试)

* （2）可用属性：

**coef_**:训练后的输入端模型系数，如果label有两个，即y值有两列。那么是一个2D的array

**intercept_**: 截距

* （3）可用的methods:

**fit(X,y,sample_weight=None):** 

X: array, 稀疏矩阵 [n_samples,n_features] 

y:  array  [n_samples, n_targets] 

**sample_weight**: 权重 array  [n_samples] 

**get_params(deep=True)：** 返回对regressor 的设置值

**predict(X):** 预测 

**score(X, y, sample_weight=None)：**，评估基于 R^2值，其结果等于

![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/linear_7.png?raw=true)

# 参考资料
[黑桃--线性回归](https://github.com/Heitao5200/Heitao5200_MachineLearning/blob/master/LR/LinearRegression.md)

[爖--线性回归](https://note.youdao.com/share/?id=981825c617d47c10f4e0c373e8b7bfff&type=note#/)

YYT的jupyter notebook

[Datawhale--线性回归](https://blog.csdn.net/Datawhale/article/details/82931967)

[李烨老师--机器学习](https://gitbook.cn/gitchat/column/5ad70dea9a722231b25ddbf8/topic/5b20586fe6a93576476f6b19)











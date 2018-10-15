
# 逻辑回归
## 一、定义
**逻辑回归又称为对数几率回归，属于处理二分类问题的广义的线性回归模型，即将数据集放入线性表达式后带入Sigmoid函数得到其属于不同类别的概率并以此确定所属类别。**
* 逻辑回归是在数据服从伯努利分布的假设下，通过极大似然的方法，运用梯度下降法来求解参数，从而达到将数据二分类的目的。
* 假设条件：（1）数据服从伯努利分布（例如抛硬币）；（2）假设样本为正的概论 p 为一个 Sigmoid 函数。 

（1）逻辑回归对样本概率的估计类似线性回归，也是计算出样本的一系列权重，然后将该权重线性加和之后输入到sigmoid函数中，进而计算出一个概率值。

![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/logistic_1.jpg?raw=true)

（2）sigmoid函数将θTx的值域从R映射到 (0, 1)，从而表示发生事件的概率值，然后根据计算出来的概率值p>=0.5归为1，P<0.5归为0。

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;P=h_{\theta&space;}(x)=\frac{1}{1&plus;e^{\theta&space;^{T}x}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;P=h_{\theta&space;}(x)=\frac{1}{1&plus;e^{\theta&space;^{T}x}}" title="P=h_{\theta }(x)=\frac{1}{1+e^{\theta ^{T}x}}" /></a>

## 二、损失函数
（1）我们既然是通过sigmoid函数的值来进行概率预测的，那么我们的目标就应该是找出一组权重参数θ，能够对于正样本使得sigmoid函数有一个高的输出值，而对于负样本有一个低的输出。我们可以通过计算损失函数来逐步达到这一的目标。

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;L(\theta&space;)=\prod&space;_{i=1}^{m}P(y|x;\theta&space;)&space;=\prod&space;_{i=1}^{m}(h_{\theta&space;}(x_{i})^{y_{i}}(1-h_{\theta&space;}(x_{i}))^{(1-y^{i})})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;L(\theta&space;)=\prod&space;_{i=1}^{m}P(y|x;\theta&space;)&space;=\prod&space;_{i=1}^{m}(h_{\theta&space;}(x_{i})^{y_{i}}(1-h_{\theta&space;}(x_{i}))^{(1-y^{i})})" title="L(\theta )=\prod _{i=1}^{m}P(y|x;\theta ) =\prod _{i=1}^{m}(h_{\theta }(x_{i})^{y_{i}}(1-h_{\theta }(x_{i}))^{(1-y^{i})})" /></a>

（2）为了便于计算，取对数。

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;logL(\theta&space;)=\sum&space;_{i=1}^{m}log(h_{\theta&space;}(x_{i})^{y_{i}}(1-h_{\theta&space;}(x_{i}))^{(1-y^{i})})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;logL(\theta&space;)=\sum&space;_{i=1}^{m}log(h_{\theta&space;}(x_{i})^{y_{i}}(1-h_{\theta&space;}(x_{i}))^{(1-y^{i})})" title="logL(\theta )=\sum _{i=1}^{m}log(h_{\theta }(x_{i})^{y_{i}}(1-h_{\theta }(x_{i}))^{(1-y^{i})})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;=\sum&space;_{i=1}^{m}[y_{i}log(h_{\theta&space;}(x_{i})&plus;(1-y_{i})log(1-h_{\theta&space;}(x_{i}))]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;=\sum&space;_{i=1}^{m}[y_{i}log(h_{\theta&space;}(x_{i})&plus;(1-y_{i})log(1-h_{\theta&space;}(x_{i}))]" title="=\sum _{i=1}^{m}[y_{i}log(h_{\theta }(x_{i})+(1-y_{i})log(1-h_{\theta }(x_{i}))]" /></a>

（3）通过最小化负的对数似然函数得到最终损失函数表达式，被称为交叉熵(cross entropy) loss函数：

 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;J(\theta&space;)=-\frac{1}{m}\sum&space;_{i=1}^{m}[y_{i}log(h_{\theta&space;}(x_{i})&plus;(1-y_{i})log(1-h_{\theta&space;}(x_{i}))]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;J(\theta&space;)=-\frac{1}{m}\sum&space;_{i=1}^{m}[y_{i}log(h_{\theta&space;}(x_{i})&plus;(1-y_{i})log(1-h_{\theta&space;}(x_{i}))]" title="J(\theta )=-\frac{1}{m}\sum _{i=1}^{m}[y_{i}log(h_{\theta }(x_{i})+(1-y_{i})log(1-h_{\theta }(x_{i}))]" /></a>
 
 
 
## 三、参数求解方法

**极大似然函数无法直接求解，一般是通过对该函数进行梯度下降来不断逼近其最优解。**

* 批梯度下降：会获得全局最优解，缺点是在更新每个参数的时候需要遍历所有的数据，计算量会很大，并且会有很多的冗余计算，导致的结果是当数据量大的时候，每个参数的更新都会很慢。

* 随机梯度下降：仅选取一个样本来求梯度，是以高方差频繁更新，优点是会跳到新的和潜在更好的局部最优解，缺点是使得收敛到局部最优解的过程更加的复杂。

* 小批量梯度下降：结合了批梯度下降和随机梯度下降的优点，每次更新的时候使用 n 个样本。减少了参数更新的次数，可以达到更加稳定收敛结果，一般在深度学习当中我们采用这种方法。

## 多分类问题softmax
softmax其实是Logistic的推广到多类别分类应用中，不需建立多个二分类分类器来实现多类别分类。softmax分类器的思想很简单，对于一个新的样本，softmax回归模型对于每一类都先计算出一个分数，然后通过softmax函数得出一个概率值，根据最终的概率值来确定属于哪一类。

![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/logistic_2.jpg?raw=true)

和逻辑回归一样，得到loss函数为：

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;J(\theta&space;)=-\frac{1}{m}[\sum&space;_{i=1}^{m}\sum&space;_{j=1}^{k}1.[{y^{(i)=j}]log\frac{e^{\theta&space;_{T}^{j}x^{(i)}}}{\sum&space;_{l=1}^{k}e\theta&space;_{l}^{T}x^{(i)}}}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;J(\theta&space;)=-\frac{1}{m}[\sum&space;_{i=1}^{m}\sum&space;_{j=1}^{k}1.[{y^{(i)=j}]log\frac{e^{\theta&space;_{T}^{j}x^{(i)}}}{\sum&space;_{l=1}^{k}e\theta&space;_{l}^{T}x^{(i)}}}]" title="J(\theta )=-\frac{1}{m}[\sum _{i=1}^{m}\sum _{j=1}^{k}1.[{y^{(i)=j}]log\frac{e^{\theta _{T}^{j}x^{(i)}}}{\sum _{l=1}^{k}e\theta _{l}^{T}x^{(i)}}}]" /></a>

其中的1.[.]是一个指示性函数，即当大括号中的值为真时，该函数的结果就为1，否则其结果就为0。

![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/logistic_3.jpg?raw=true)

* 看下面的推到我们可以发现，对每一个参数减去一个参数，最后的结果没有影响。其实softmax 回归中对参数的最优化解不只一个，每当求得一个优化参数时，如果将这个参数的每一项都减掉同一个数，其得到的损失函数值也是一样的，这说明解不是唯一的。之所以会出现这样的现象，因为损失函数不是严格非凸的，也就是说在局部最小值点附近是一个”平坦”的，所以在这个参数附近的值都是一样的了。为避免出现这样的情况，加入正则项

![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/logistic_4.jpg?raw=true)

### 逻辑回归与Softmax回归的联系
![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/logistic_5.jpg?raw=true)

## 四、逻辑回归的优缺点
* 优点：

（1）形式简单，模型的可解释性非常好。从特征的权重可以看到不同的特征对最后结果的影响。

（2）模型效果不错。在工程上是可以接受的（作为 baseline），如果特征工程做的好，效果不会太差，并且特征工程可以并行开发，大大加快开发的速度。

（3）训练速度较快。分类的时候，计算量仅仅只和特征的数目相关。并且逻辑回归的分布式优化 SGD 发展比较成熟。

（4）方便调整输出结果，通过调整阈值的方式。

（5）既可以得到分类结果也可以得到类别的概率值。

* 缺点：

（1）准确率欠佳。因为形式非常的简单，而现实中的数据非常复杂，因此，很难达到很高的准确性。

（2）很难处理数据不平衡的问题。举个例子：如果我们对于一个正负样本非常不平衡的问题比如正负样本比 10000:1。我们把所有样本都预测为正也能使损失函数的值比较小。但是作为一个分类器，它对正负样本的区分能力不会很好。

（3）对模型中自变量[多重共线性](https://www.jianshu.com/p/ef1b27b8aee0)较为敏感

（4）sigmoid函数两端斜率小，模型输出的概率值变化小，中间段斜率大，概率变化大。这导致特征某些区间的数值变化对概率的影响较大。

## 五、常见面试题
（1）逻辑回归中为什么使用对数损失而不用平方损失？

 * 众所周知，线性模型是平方损失函数。对于逻辑回归，这里所说的对数损失和极大似然是相同的。 
 
 * 不使用平方损失的原因是，在使用 Sigmoid 函数作为正样本的概率时，同时将平方损失作为损失函数，这时所构造出来的损失函数是非凸的，不容易求解，容易得到其局部最优解。 而如果使用极大似然，其目标函数就是对数似然函数，该损失函数是关于未知参数的高阶连续可导的凸函数，便于求其全局最优解。

（2）为什么选择sigmoid函数做分类？

* sigmoid函数输出值在0, 1之间；
* sigmoid函数单调递增；（对于大多数线性分类器，响应值<w, x> （w 和 x 的内积） 代表了数据 x 属于正类（y = 1）的 置信度，我们需要<w, x> 越大，这个数据属于正类的可能性越大，<w,x> 越小，属于反类的可能性越大。）


（3）逻辑回归在训练的过程当中，如果有很多的特征高度相关或者说有一个特征重复了很多遍，会造成怎样的影响？

* 如果在损失函数最终收敛的情况下，其实就算有很多特征高度相关也不会影响分类器的效果。 但是对特征本身来说的话，假设只有一个特征，在不考虑采样的情况下，你现在将它重复 N 遍。训练以后完以后，数据还是这么多，但是这个特征本身重复了 N 遍，实质上将原来的特征分成了 N 份，每一个特征都是原来特征权重值的百分之一。

## 六、Sklearn库中Logistic Regression函数各个参数总结

![upload_picture](https://github.com/wangjiaxin24/machine_learning-52/blob/master/upload_picture/logistic_7.jpg?raw=true)

# 参考资料
[ML--广义线性回归(线性回归、逻辑回归)](https://blog.csdn.net/jiebanmin0773/article/details/82962182)

[逻辑回归算法面经](https://mp.weixin.qq.com/s__biz=MzI4Mzc5NDk4MA==&mid=2247484688&idx=6&sn=cdff744e9db787578552416f4dcf222b&chksm=eb840e5bdcf3874d4ad546361dc4247287b528b6cb4988dda3837d5a6bfb73a7961aabbab32a&mpshare=1&scene=1&srcid=1011bwPIPARtKOq4hzUPpnpR#rd])

[爖的有道云笔记](https://note.youdao.com/share/?id=3736895c09a621e8c3e0b430d2ead239&type=note#/)

[Logistic and Softmax Regression](https://blog.csdn.net/danieljianfeng/article/details/41901063)

# 支持向量机
支持向量机（Support Vector Machines, SVM）是一种二分类模型。它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机；支持向量机还包括核技巧，这使其成为实质上的非线性分类器。

SVM 的学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。

SVM 的最优化算法是求解凸二次规划的最优化算法。

## 什么是支持向量

训练数据集中与分离超平面距离最近的样本点的实例称为支持向量

更通俗的解释：
数据集种的某些点，位置比较特殊。比如 x+y-2=0 这条直线，假设出现在直线上方的样本记为 A 类，下方的记为 B 类。
在寻找找这条直线的时候，一般只需看两类数据，它们各自最靠近划分直线的那些点，而其他的点起不了决定作用。
这些点就是所谓的“支持点”，在数学中，这些点称为向量，所以更正式的名称为“支持向量”。


## 支持向量机的分类
(1) 线性可分支持向量机

当训练数据线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机，又称硬间隔支持向量机。

(2) 线性支持向量机

当训练数据接近线性可分时，通过软间隔最大化，学习一个线性分类器，即线性支持向量机，又称软间隔支持向量机。

(3) 非线性支持向量机
当训练数据线性不可分时，通过使用核技巧及软间隔最大化，学习非线性支持向量机。

## 核函数与核技巧

核函数表示将输入从输入空间映射到特征空间后得到的特征向量之间的内积

# 最大间隔超平面背后的原理
相当于在最小化权重时对训练误差进行了约束——对比 L2 范数正则化，则是在最小化训练误差时，对权重进行约束

相当于限制了模型复杂度——在一定程度上防止过拟合，具有更强的泛化能力

## 支持向量机推导

当训练数据线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机，又称硬间隔支持向量机。
线性 SVM 的推导分为两部分

1: 如何根据间隔最大化的目标导出 SVM 的标准问题；
2: 拉格朗日乘子法对偶问题的求解过程.


(1) 从“函数间隔”到“几何间隔”

给定训练集T和超平面(w,b)，定义函数间隔γ^

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\hat{\gamma}&=\underset{i=1,\cdots,N}{\min},y_i(wx_i&plus;b)&space;\&space;&=\underset{i=1,\cdots,N}{\min},\hat{\gamma}_i\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;\hat{\gamma}&=\underset{i=1,\cdots,N}{\min},y_i(wx_i&plus;b)&space;\&space;&=\underset{i=1,\cdots,N}{\min},\hat{\gamma}_i\end{aligned}" title="\begin{aligned} \hat{\gamma}&=\underset{i=1,\cdots,N}{\min},y_i(wx_i+b) \ &=\underset{i=1,\cdots,N}{\min},\hat{\gamma}_i\end{aligned}" /></a>


对 w 作规范化，使函数间隔成为几何间隔γ

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\gamma&=\underset{i=1,\cdots,N}{\min},y_i(\frac{w}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}}x_i&plus;\frac{b}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}})\&space;&=\underset{i=1,\cdots,N}{\min},\frac{\gamma_i}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}}&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;\gamma&=\underset{i=1,\cdots,N}{\min},y_i(\frac{w}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}}x_i&plus;\frac{b}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}})\&space;&=\underset{i=1,\cdots,N}{\min},\frac{\gamma_i}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}}&space;\end{aligned}" title="\begin{aligned} \gamma&=\underset{i=1,\cdots,N}{\min},y_i(\frac{w}{{\color{Red} \left | w \right |}}x_i+\frac{b}{{\color{Red} \left | w \right |}})\ &=\underset{i=1,\cdots,N}{\min},\frac{\gamma_i}{{\color{Red} \left | w \right |}} \end{aligned}" /></a>

(2) 最大化几何间隔

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;&\underset{w,b}{\max}&space;\quad{\color{Red}&space;\frac{\hat{\gamma}}{\left&space;|&space;w&space;\right&space;|}}&space;&&space;\mathrm{s.t.}\quad,&space;y_i(wx_i&plus;b)&space;\geq&space;{\color{Red}&space;\hat{\gamma}},\quad&space;i=1,2,\cdots,N&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;&\underset{w,b}{\max}&space;\quad{\color{Red}&space;\frac{\hat{\gamma}}{\left&space;|&space;w&space;\right&space;|}}&space;&&space;\mathrm{s.t.}\quad,&space;y_i(wx_i&plus;b)&space;\geq&space;{\color{Red}&space;\hat{\gamma}},\quad&space;i=1,2,\cdots,N&space;\end{aligned}" title="\begin{aligned} &\underset{w,b}{\max} \quad{\color{Red} \frac{\hat{\gamma}}{\left | w \right |}} & \mathrm{s.t.}\quad, y_i(wx_i+b) \geq {\color{Red} \hat{\gamma}},\quad i=1,2,\cdots,N \end{aligned}" /></a>

函数间隔γ^的取值不会影响最终的超平面(w,b)：取γ^=1；又最大化 1/||w|| 等价于最小化1/2*||w||^2，于是有

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;&{\color{Red}&space;\underset{w,b}{\max}&space;}&space;\quad\frac{\hat{\gamma}}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}}&space;\&space;&&space;\mathrm{s.t.}\quad,&space;y_i(wx_i&plus;b)&space;\geq&space;\hat{\gamma}_i,\quad&space;i=1,2,\cdots,N&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;&{\color{Red}&space;\underset{w,b}{\max}&space;}&space;\quad\frac{\hat{\gamma}}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}}&space;\&space;&&space;\mathrm{s.t.}\quad,&space;y_i(wx_i&plus;b)&space;\geq&space;\hat{\gamma}_i,\quad&space;i=1,2,\cdots,N&space;\end{aligned}" title="\begin{aligned} &{\color{Red} \underset{w,b}{\max} } \quad\frac{\hat{\gamma}}{{\color{Red} \left | w \right |}} \ & \mathrm{s.t.}\quad, y_i(wx_i+b) \geq \hat{\gamma}_i,\quad i=1,2,\cdots,N \end{aligned}" /></a>


为什么令γ^=1？

——比例改变(ω,b)，超平面不会改变，但函数间隔γ^会成比例改变，因此可以通过等比例改变(ω,b)使函数间隔γ^=1


该约束最优化问题即为线性支持向量机的标准问题——这是一个凸二次优化问题，可以使用商业 QP 代码完成。

理论上，线性 SVM 的问题已经解决了；但在高等数学中，带约束的最优化问题还可以用另一种方法求解——拉格朗日乘子法。该方法的优点一是更容易求解，而是自然引入核函数，进而推广到非线性的情况。

(3) SVM 对偶算法的推导
1. 构建拉格朗日函数

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;L(w,b,{\color{Red}&space;\alpha})=&\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red}&space;\alpha_i}[y_i(w^Tx_i&plus;b)-1]\&space;&{\color{Red}&space;\alpha_i&space;\geq&space;0},\quad&space;i=1,2,\cdots,N&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;L(w,b,{\color{Red}&space;\alpha})=&\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red}&space;\alpha_i}[y_i(w^Tx_i&plus;b)-1]\&space;&{\color{Red}&space;\alpha_i&space;\geq&space;0},\quad&space;i=1,2,\cdots,N&space;\end{aligned}" title="\begin{aligned} L(w,b,{\color{Red} \alpha})=&\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red} \alpha_i}[y_i(w^Tx_i+b)-1]\ &{\color{Red} \alpha_i \geq 0},\quad i=1,2,\cdots,N \end{aligned}" /></a>

2. 标准问题是求极小极大问题

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;{\color{Red}&space;\underset{w,b}{\min}}&space;{\color{Blue}&space;\underset{\alpha}{\max}}&space;L(w,b,\alpha)&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;{\color{Red}&space;\underset{w,b}{\min}}&space;{\color{Blue}&space;\underset{\alpha}{\max}}&space;L(w,b,\alpha)&space;\end{aligned}" title="\begin{aligned} {\color{Red} \underset{w,b}{\min}} {\color{Blue} \underset{\alpha}{\max}} L(w,b,\alpha) \end{aligned}" /></a>

**对偶问题为：**

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;{\color{Blue}&space;\underset{\alpha}{\max}}&space;{\color{Red}&space;\underset{w,b}{\min}}&space;L(w,b,\alpha)&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;{\color{Blue}&space;\underset{\alpha}{\max}}&space;{\color{Red}&space;\underset{w,b}{\min}}&space;L(w,b,\alpha)&space;\end{aligned}" title="\begin{aligned} {\color{Blue} \underset{\alpha}{\max}} {\color{Red} \underset{w,b}{\min}} L(w,b,\alpha) \end{aligned}" /></a>

3. 求 L 对 (w,b) 的极小

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\mathrm{set}\quad&space;\frac{\partial&space;L}{\partial&space;w}=0&space;;;&\Rightarrow;&space;w-\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i&space;x_i}=0\&space;&\Rightarrow;&space;w=\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i&space;x_i}&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;\mathrm{set}\quad&space;\frac{\partial&space;L}{\partial&space;w}=0&space;;;&\Rightarrow;&space;w-\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i&space;x_i}=0\&space;&\Rightarrow;&space;w=\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i&space;x_i}&space;\end{aligned}" title="\begin{aligned} \mathrm{set}\quad \frac{\partial L}{\partial w}=0 ;;&\Rightarrow; w-\sum_{i=1}^N {\color{Red} \alpha_i y_i x_i}=0\ &\Rightarrow; w=\sum_{i=1}^N {\color{Red} \alpha_i y_i x_i} \end{aligned}" /></a>


<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\mathrm{set}\quad&space;\frac{\partial&space;L}{\partial&space;b}=0&space;;;&\Rightarrow;&space;\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i}=0&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;\mathrm{set}\quad&space;\frac{\partial&space;L}{\partial&space;b}=0&space;;;&\Rightarrow;&space;\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i}=0&space;\end{aligned}" title="\begin{aligned} \mathrm{set}\quad \frac{\partial L}{\partial b}=0 ;;&\Rightarrow; \sum_{i=1}^N {\color{Red} \alpha_i y_i}=0 \end{aligned}" /></a>

4. 结果代入L，有：

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;L(w,b,{\color{Red}&space;\alpha})&space;&=\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red}&space;\alpha_i}[y_i(w^Tx_i&plus;b)-1]\&space;&=\frac{1}{2}w^Tw-w^T\sum_{i=1}^N&space;\alpha_iy_ix_i-b\sum_{i=1}^N&space;\alpha_iy_i&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=\frac{1}{2}w^Tw-w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=-\frac{1}{2}w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N&space;\alpha_i\alpha_j\cdot&space;y_iy_j\cdot&space;{\color{Red}&space;x_i^Tx_j}&plus;\sum_{i=1}^N&space;\alpha_i&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;L(w,b,{\color{Red}&space;\alpha})&space;&=\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red}&space;\alpha_i}[y_i(w^Tx_i&plus;b)-1]\&space;&=\frac{1}{2}w^Tw-w^T\sum_{i=1}^N&space;\alpha_iy_ix_i-b\sum_{i=1}^N&space;\alpha_iy_i&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=\frac{1}{2}w^Tw-w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=-\frac{1}{2}w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N&space;\alpha_i\alpha_j\cdot&space;y_iy_j\cdot&space;{\color{Red}&space;x_i^Tx_j}&plus;\sum_{i=1}^N&space;\alpha_i&space;\end{aligned}" title="\begin{aligned} L(w,b,{\color{Red} \alpha}) &=\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red} \alpha_i}[y_i(w^Tx_i+b)-1]\ &=\frac{1}{2}w^Tw-w^T\sum_{i=1}^N \alpha_iy_ix_i-b\sum_{i=1}^N \alpha_iy_i+\sum_{i=1}^N \alpha_i\ &=\frac{1}{2}w^Tw-w^Tw+\sum_{i=1}^N \alpha_i\ &=-\frac{1}{2}w^Tw+\sum_{i=1}^N \alpha_i\ &=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_j\cdot y_iy_j\cdot {\color{Red} x_i^Tx_j}+\sum_{i=1}^N \alpha_i \end{aligned}" /></a>

即

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;L(w,b,{\color{Red}&space;\alpha})&space;&=\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red}&space;\alpha_i}[y_i(w^Tx_i&plus;b)-1]\&space;&=\frac{1}{2}w^Tw-w^T\sum_{i=1}^N&space;\alpha_iy_ix_i-b\sum_{i=1}^N&space;\alpha_iy_i&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=\frac{1}{2}w^Tw-w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=-\frac{1}{2}w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N&space;\alpha_i\alpha_j\cdot&space;y_iy_j\cdot&space;{\color{Red}&space;x_i^Tx_j}&plus;\sum_{i=1}^N&space;\alpha_i&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;L(w,b,{\color{Red}&space;\alpha})&space;&=\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red}&space;\alpha_i}[y_i(w^Tx_i&plus;b)-1]\&space;&=\frac{1}{2}w^Tw-w^T\sum_{i=1}^N&space;\alpha_iy_ix_i-b\sum_{i=1}^N&space;\alpha_iy_i&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=\frac{1}{2}w^Tw-w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=-\frac{1}{2}w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\&space;&=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N&space;\alpha_i\alpha_j\cdot&space;y_iy_j\cdot&space;{\color{Red}&space;x_i^Tx_j}&plus;\sum_{i=1}^N&space;\alpha_i&space;\end{aligned}" title="\begin{aligned} L(w,b,{\color{Red} \alpha}) &=\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red} \alpha_i}[y_i(w^Tx_i+b)-1]\ &=\frac{1}{2}w^Tw-w^T\sum_{i=1}^N \alpha_iy_ix_i-b\sum_{i=1}^N \alpha_iy_i+\sum_{i=1}^N \alpha_i\ &=\frac{1}{2}w^Tw-w^Tw+\sum_{i=1}^N \alpha_i\ &=-\frac{1}{2}w^Tw+\sum_{i=1}^N \alpha_i\ &=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_j\cdot y_iy_j\cdot {\color{Red} x_i^Tx_j}+\sum_{i=1}^N \alpha_i \end{aligned}" /></a>

5. 求 L 对 α 的极大，即




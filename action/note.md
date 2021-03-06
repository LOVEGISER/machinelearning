# 常用的机器学习算法
		
| 监督学习 | 非监督学习          |
| ----------| ---------|
| k邻近值算法      | k均值|
| 朴素贝叶斯算法      | DBSCAN|
| 支持向量机      | 最大期望算法|
| 决策树      | parzen窗设计|
| 线性回归      | |
| 局部加权线性回归      |  |
 | Ridge回归      |  |
 | Lasso回归      |  |
 # 如何选择机器学习算法
  1：如果想要预测目标变量的值， 则可以选择监督学习算法，否则可以选择无监督学习算法。  
 2：确定选择监督学习算法之后，需要进一步确定目标变量类型，如果目标变量是离散型，
 如是/否、1/2/3、― 冗或者红/黄/黑等，则可以选择`分类器算法`；如果目标变量是连续型的数值，
 如0.0~ 100.00、-999~999或者+00~-00等 ，则需要选择`回归算法`。  
 3：如果不想预测目标变量的值，则可以选择无监督学习算法。进一步分析是否需要将数据划分
为离散的组。如果这是唯一的需求，则使用`聚类算法`；如果还需要估计数据与每个分组的相似程
度 ，则需要使用`密度估计算法`。

##  k-近邻算法
简单地说，谷近邻算法采用测量不同特征值之间的距离方法进行分类。  
优 点 ：精度高、对异常值不敏感、无数据输入假定。  
缺点：计算复杂度高、空间复杂度高。    
适用数据范围：数值型和标称型。   
### 工作原理是：
存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据
与所属分类的对应关系。输人没有标签的新数据后， 将新数据的每个特征与样本集中数据对应的
特征进行比较，然后算法提取样本集中特征最相似数据（最 近 邻 ）的分类标签。

##  决策树
在构造决策树时，我们需要解决的第一个问题就是，当前数据集上哪个特征在划分数据分类
时起决定性作用。为了找到决定性的特征，划分出最好的结果，我们必须评估每个特征。完成测
试之后，原始数据集就被划分为几个数据子集。这些数据子集会分布在第一个决策点的所有分支
上 。 如果某个分支下的数据属于同一类型，则当前无需阅读的垃圾邮件已经正确地划分数据分类，
无需进一步对数据集进行分割。如果数据子集内的数据不属于同一类型，则需要重复划分数据子
集的过程。如何划分数据子集的算法和划分原始数据集的方法相同，直到所有具有相同类型的数
据均在一个数据子集内
### 决策树的一般流程
(1)收集数据：可以使用任何方法。  
(2)准备数据：`树构造算法只适用于标称型数据，因此数值型数据必须离散化`。  
(3)分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。  
(4)训练算法：构造树的数据结构。  
(5)测试算法：使用经验树计算错误率。  
(6)使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据
的内在含义。  
 ### 信息増益
在划分数据集之前之后信息发生的变化称为信息增益， 知道如何计算信息增益，我们就可以
计算每个特征值划分数据集获得的信息增益，`获得信息增益最高的特征就是最好的选择`  
### 熵
熵是集合信息的度量的一种方式,度量数据集的无序程度.熵定义为信息的期望值。
 ### 划分数据集
上节我们学习了如何度量数据集的无序程度，分类算法除了需要测量信息熵，还需要划分数
据集，度量划分数据集的熵，以便判断当前是否正确地划分了数据集。我们将对每个特征划分数
据集的结果计算一次信息熵，然后判断按照哪个特征划分数据集是最好的划分方式。

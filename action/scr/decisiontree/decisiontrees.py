
from math import log
import operator
import matplotlib.pyplot as plt
import pickle
#计算香浓商-数学期望，即就是特征出现的概率 l(xi)=-p(xi)*log(p(xi))
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2) #log base 2
    return shannonEnt

#模拟构造数据
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'no'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

dataSet, labels = createDataSet()
print(calcShannonEnt(dataSet))

#安装给定特殊划分数据集dataSet:要划分的数据集 ，axis：按照那个维度划分，value：划分标准
def spliteDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec [:axis]
            reducedFeatVec.extend(featVec [axis+1:]);
            retDataSet.append(reducedFeatVec)
    return retDataSet

print(spliteDataSet(dataSet,0,1))

#遍历整个数据集，循环计算香农熵和spliteDataSet函数，找到最好的特征划分方式。熵计算将会告诉我们如何划分数据集是最好的数据组织方式。
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    #Entropy ：熵,计算数据集的香浓熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0 ; bestFeature = -1
    for i in range (numFeature):
        #获取第i个维度的所有特征值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntroy = 0.0
        for value in uniqueVals:
            #对第i个维度的特征值做数据集划分
            subDataSet = spliteDataSet(dataSet,i,value)
            prob = len(subDataSet) / float (len(dataSet))
            #计算划分完数据集之后的香浓熵并且加和(即信息增益-信息增益是熵的减少或者是数据无序度的减少)
            newEntroy += prob * calcShannonEnt(subDataSet)
        #newEntroy ： 第i个维度熵的数据经过特征集划分后的香浓熵
        infoGain = baseEntropy - newEntroy
        #比较所有特征中的信息增益，返回最好特征划分的索引值
        if(infoGain > bestInfoGain):
            baseEntropy = infoGain
            bestFeature = i
    return  bestFeature

print(chooseBestFeatureToSplit(dataSet))

# 如果数据集已经处理了所有属性， 但是类标签依然不是唯一
# 的 ，此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决
# 定该叶子节点的分类。
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
#创建决策树
#
def createTree(dataSet,labels):
    #
    classList = [example[-1] for example in dataSet]
    #划分类别。类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList) :
        return  classList [0]
    #使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分
    #遍历完所有特征，返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #获取当下数据最好的分类属性/维度
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #获取最好维度所对应的特征标签
    bestFeatLabel = labels[bestFeat]
    #myTree为决策树
    myTree = {bestFeatLabel:{}}
    #删除当前维度特征，这样每次计算都会消除 掉一个属性/维度
    del[labels[bestFeat]]
    #获取当前最优决策/分类维度的所有特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    #遍历最佳维度下的每个特征值，并却根据该特征值分类数据
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(spliteDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


#print()

# 依靠训练数据构造了决策树之后，我们可以将它用于实际数据的分类。在执行数据分类时，
# 需要决策树以及用于构造树的标签向量。然后，程序比较测试数据与决策树上的数值，递归执行
# 该过程直到进人叶子节点；最后将测试数据定义为叶子节点所属的类型。
def classify(inputTree,featLabels,testVec):
    #树的父节点
    firstStr = list(inputTree.keys())[0]
    #决策树的子节点数据
    secondDict = inputTree[firstStr]
    #父节点锁对应的索引
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    #如果secondDict的子节点是dict ，说明未完全分类，否则直接返回
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel



#mytree = createTree(dataSet,['no surfacing','flippers']);
#print(classify(mytree,labels,[1,0]))

#存储决策树
def storeTree(inputTree,fileName):
    fw = open(fileName,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
#反序列化决策树
def grabTree(fileName):
     fr = open(fileName,'rb')
     return  pickle.load(fr)
# mytree = createTree(dataSet,['no surfacing','flippers']);
# storeTree(mytree,'mytree.txt')
# print(grabTree('mytree.txt'))

#利用决策树判断应该佩戴的眼镜
# 返回一个文件对象
f = open("lenses.txt")
line = f.readline()  # 调用文件的 readline()方法
lenses = []
while line:
    line = f.readline()
    lenses.append(line.strip('\n').split('\t'))
f.close()

lensesLabels = ['age','prescript','astigmatic','tearTate']
lensesTree = createTree(lenses[1:23],lensesLabels)
print(lensesTree)
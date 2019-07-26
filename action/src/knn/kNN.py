#K值最邻近算法

from numpy import *
import operator

def createDataSet():
    group =  array( [[1.0,1.1 ],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group , lables

#inX 用于预测的数据，dataSet用于计算数据集，lables和dataSet 对应的类型 ，k：返回多少个
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    #从距离排名前k个数据中取值，计算每个类别出现的次数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #按照次数倒排
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回距离最近的点
    return sortedClassCount[0][0]

#数据归一化函数(newValue=(oldValue-min)/(max-min))
def autoNorm(dataSet):
    #dataSet.min(0) 中的参数0使得函数可以从列中选取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet / tile(ranges,(m,1))
    return  normDataSet,ranges,minVals



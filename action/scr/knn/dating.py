import kNN
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt

def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[3].isdigit()):
            classLabelVector.append(int(listFromLine[3]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[3]))
        index += 1

    return returnMat,classLabelVector


dataSetMat,classLabelVector = file2matrix("resources/datingTestSet.txt")
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataSetMat[:,1],dataSetMat[:,2])
# plt.show()

# normDataSet,range,minVals = kNN.autoNorm(dataSetMat)
# print(normDataSet)

def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = dataSetMat,classLabelVector = file2matrix("resources/datingTestSet.txt")       #load data setfrom file
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range (numTestVecs):
        classifierResult = kNN.classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if( classifierResult != datingLabels[i]):errorCount+=1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)




def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    # percentTats = float(input("percentage of time spent playing video games?"))
    # ffMiles = float(input("frequent flier miles earned per year?"))
    # iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('resources/datingTestSet2.txt')
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    inArr = array([9289,9.666576,1.370330, ])
    classifierResult = kNN.classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: %s" % resultList[classifierResult - 1])

#print(classifyPerson())

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('resources/trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('resources/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('resources/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('resources/testDigits/%s' % fileNameStr)
        classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


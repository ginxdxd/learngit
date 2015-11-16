# -*- coding: utf-8 -*-
# 上句必须写在最开头。
__author__ = 'quanxiandeng'
#!/usr/bin/env python


# 以下代码为个人参考 andrew ng 神经网络部分

'''
参考资料
http://blog.csdn.net/akunainiannian/article/details/40073903  BP神经网络Python实现（带有”增加充量项“）
http://www.mamicode.com/info-detail-671452.html  神经网络和反向传播算法推导



'''
import numpy as np
import pandas as pd
import math

'''
    useful features : Pclass  Sex   Age  SibSP  Parch   Fare  Embarked
    useless features : PassengerId Name Ticket Cabin
'''

def main():
    # check = np.array([[2,3], [4,5]])
    # check2 = check * check
    data, label = load_data()    # 得到训练集
    testdata = load_testdata()   # 得到测试集
    # data = data[:100]
    datalist = [data, testdata]
    CleanData(datalist)

    hiddenLayer = 1              # 隐藏层数
    hiddenUnits = data.shape[1]  # 每层激励单元数  number of neurons
    outputUnits = 1              # 输出单元数
    # learningrate = math.pow(10, -1)
    learningrate = 0.005
    epson = math.pow(10, -6)  # 0.0001
    lumta = 0.5
    costsum = 0.0

    label = label.as_matrix()    # dataframe to matrix
    data = data.as_matrix()
    testdata = testdata.as_matrix()

    print type(data)
    finaltheta = Neur_net(data, label, hiddenLayer, hiddenUnits, outputUnits, learningrate, epson, costsum, lumta)
    hit = hittarget(test(finaltheta, data, label), label)
    finaloutput = test(finaltheta, testdata)
#  end main

def load_data():
    path = '/Users/quanxiandeng/Downloads/Data/titanic/train.csv'
    traindata = pd.read_csv(path)
    label = traindata['Survived']
    return traindata, label
#  end loadData

def load_testdata():
    path = '/Users/quanxiandeng/Downloads/Data/titanic/test.csv'
    testData = pd.read_csv(path)
    return testData
#  end load testdata

def Neur_net(data, label, hl, hiddenUnit, outputUnits, learningrate, epson, costsum, lumta):
    m, n = np.shape(data)
    thetalist = []
    dthetalist = []
    getTheta(thetalist, n, outputUnits, hiddenUnit, hl)
    difference = np.zeros_like(label, dtype=float)
    costsum = train(m, data, hl, thetalist, label, costsum, difference, dthetalist, lumta)

    # jtheta = 0.5 * costsum
    Regularization = regularization(thetalist)
    jtheta = ((-1.0) / m) * costsum + 0.5 * (lumta / m) * Regularization

    gradient_desc(thetalist, dthetalist, learningrate)
    costsum = train(m, data, hl, thetalist, label, costsum, difference, dthetalist, lumta)

    # newjtheta = 0.5 * costsum
    Regularization = regularization(thetalist)
    newjtheta = ((-1.0) / m) * costsum + 0.5 * (lumta / m) * Regularization

    time = 1
    while(jtheta - newjtheta > epson):
        jtheta = newjtheta
        gradient_desc(thetalist, dthetalist, learningrate)
        costsum = train(m, data, hl, thetalist, label, costsum, difference, dthetalist, lumta)
        newjtheta = 0.5 * costsum
        time = time + 1
    print 'run' + str(time) + 'times'
    return thetalist
#  end Neur_net

def back_propagation(delta, thetalist, amatrix, hl):  # thetalist size: 2  amatrix size: 3
    deltamatrix = []
    deltamatrix.append(delta)
    i = len(amatrix) - 1
    while(hl > 0):
        delta = np.multiply(np.dot(thetalist[hl].T, delta), np.multiply(amatrix[i - 1].T, (1 - amatrix[i - 1].T)))   # doubt  dot 矩阵乘法  mul 对位乘法
        deltamatrix.append(delta)
        i -= 1
        hl -= 1
    return deltamatrix
#  end back_propagation

def costFunction(label, outputdata):
    return label * math.log(outputdata) + (1 - label) * math.log(1 - (outputdata))
#  end costFunction

def forward_propagation(inputdata, theta):
    z = np.dot(theta, inputdata.T)
    a = np.apply_along_axis(sigmoid_function, 1, z)
    # a = sigmoid_function(z)
    return a
#  end forward_propagation

def getTheta(thetalist, numbersOfInputFeature, outputUnits, hiddenUnit, hl):
    thetaOne = (np.random.random(size=(hiddenUnit, (numbersOfInputFeature + 1))) - 0.5) / 10.0
    thetalist.append(thetaOne)
    while hl-1 > 0:
        theta = (np.random.random(size=(hiddenUnit, (hiddenUnit + 1))) - 0.5) / 10.0
        hl -= 1
        thetalist.append(theta)
    thetalast = (np.random.random(size=(outputUnits, (hiddenUnit + 1))) - 0.5) / 10.0
    thetalist.append(thetalast)
#  end getTheta

def sigmoid_function(x):
    return 1.0 / (1 + math.exp(-x))
#  end sigmoid_function

def dSigmoid_function(x):
    return sigmoid_function(x) * (1 - sigmoid_function(x))
#  end dSigmoid_function

def gradient_desc(thetalist, dthetalist, learningrate):
    for i in range(len(thetalist)):
        thetalist[i] -= learningrate * dthetalist[i]  # 梯度方向－ 和 负方向 ＋
#  end gradient_desc

def train(m, data, hl, thetalist, label, costsum, difference, dthetalist, lumta):
    for i in range(m):
        amatrix = []          # 此样本的所有激励元 a
        deltamatrix = []      # 可有可无
        floor = 0
        xi = data[i: i+1, :]  # 抽第i个样本作为输入  1 * n matrix and n belongs to feature numbers
        # xi = data[i]
        xi = np.append([[1]], xi, axis=1)  # add bias
        amatrix.append(xi)
        while floor < hl + 1:
            a = forward_propagation(xi, thetalist[floor])
            floor += 1
            xi = np.append([[1]], [a], axis=1)
            amatrix.append(xi)

        out = amatrix[len(amatrix) - 1][:, 1:]  # 此处选择除了头一列外的矩阵
        cost = costFunction(label[i], out)
        costsum = cost + costsum                # 用来计算jtheta

        # allamatrix.append(amatrix)            # why need this ?
        difference[i] = label[i] - out
        # delta = difference[i]                   # delta y

        delta = np.multiply((out - label[i]), out * (1 - out))  # 修改后的输出层 误差项

        deltamatrix = back_propagation(delta, thetalist, amatrix, hl)  # 此样本的delta
        calculate_gradientQuo(thetalist, dthetalist, hl, amatrix, deltamatrix)

    for i in range(len(dthetalist)):
        dthetalist[i] = (1.0) / m * dthetalist[i] + lumta * thetalist[i]  # doubt?

    return costsum
    # differencesum = difference * difference
    # differencesum = differencesum.sum()
    # return differencesum
#  end train

def test(thetalist, testdata, label):
    m, n = np.shape(testdata)
    floor = 0
    output = np.zeros_like(label, dtype=int)
    for i in range(m):
        xi = testdata[i: i+1, :]
        # xi = np.hstack(1, xi)
        xi = np.append([[1]], xi, axis=1)
        while floor < len(thetalist):
            a = forward_propagation(xi, thetalist[floor])
            floor += 1
            xi = np.append([[1]], [a], axis=1)
        output[i] = a
        if a >= .5:
            output[i] = 1
        else:
            output[i] = 0

    return output
#  end test

def CleanData(datalist):
    Age_Mean = datalist[0]['Age'].median()
    datalist[1]['Fare'] = datalist[1]['Fare'].fillna(datalist[1]['Fare'].median())
    for dataset in datalist:
        dataset['Age'] = dataset['Age'].fillna(Age_Mean)
        dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 0
        dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 1
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        dataset.loc[dataset['Embarked'] == 'S', 'Embarked'] = 0
        dataset.loc[dataset['Embarked'] == 'Q', 'Embarked'] = 1
        dataset.loc[dataset['Embarked'] == 'C', 'Embarked'] = 2
        del dataset['PassengerId']
        del dataset['Name']
        del dataset['Ticket']
        del dataset['Cabin']
        # del dataset[['PassengerId', 'Name', 'Ticket', 'Cabin']]
    return datalist
#  end Cleandata

def toMatrix(obj):
    return obj.as_matrix()
#  end tomatrix

def calculate_gradientQuo(thetalist, dthetalist, hl, amatrix, deltamatrix):
    checklen = len(dthetalist)
    for l in range(hl+1):
        m, n = thetalist[l].shape
        dthetalij = np.zeros((m, n), dtype=float)
        index = len(deltamatrix)-1-l
        if index != 0:
            for j in range(n):
                for i in range(m):
                    # check = dthetalij[i][j]
                    # aJ = amatrix[l][0][j]
                    deltaI = deltamatrix[index][i+1][0]
                    dthetalij[i][j] = amatrix[l][0][j] * deltaI
        else:
            for j in range(n):
                for i in range(m):
                    # check = dthetalij[i][j]
                    # aJ = amatrix[l][0][j]
                    deltaI = deltamatrix[index]
                    dthetalij[i][j] = amatrix[l][0][j] * deltaI
        if len(dthetalist) < 2:
            dthetalist.append(dthetalij)
        else:
            dthetalist[l] += dthetalij
#  end calculate_gradientQuo

def regularization(thetalist):
    thetasum = 0.0
    for i in range(len(thetalist)):
        newtheta = thetalist[i][:, 1:] * thetalist[i][:, 1:]
        thetasum += newtheta.sum(axis=1).sum(axis=0)
    return thetasum
#  end regularization

def hittarget(output, label):
    hit = 0.0
    difference = output - label
    # check = len(difference[difference == 0])
    hit = (float(len(difference[difference == 0])) / float(len(label)))
    return hit
# end hittarget

def stochasticgd():

    return 1
# end sgd


if __name__ == '__main__':
    main()
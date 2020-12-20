import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace

def DataSet():
    dataSet = [
        [0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, -1],
        [0.243, 0.267, -1],
        [0.245, 0.057, -1],
        [0.343, 0.099, -1],
        [0.639, 0.161, -1],
        [0.657, 0.198, -1],
        [0.360, 0.370, -1],
        [0.593, 0.042, -1],
        [0.719, 0.103, -1]
    ]

    return np.array(dataSet)

def calErr(dataSet, feature, threshVal, unequal, D):
    DFlatten = D.flatten()  
    errCnt = 0
    i = 0
    if unequal == 'lt':  
        for data in dataSet:
            if (data[feature] <= threshVal and data[-1] == -1) or \
               (data[feature] > threshVal and data[-1] == 1):  
                errCnt += 1 * DFlatten[i]  
            i += 1
    else:
        for data in dataSet:
            if (data[feature] >= threshVal and data[-1] == -1) or \
               (data[feature] < threshVal and data[-1] == 1):
                errCnt += 1 * DFlatten[i]
            i += 1
    return errCnt

def buildStump(dataSet, D):
    m, n = dataSet.shape
    bestErr = np.inf
    bestStump = {}
    numSteps = 16.0  
    for i in range(n-1):                    
        rangeMin = dataSet[:, i].min()
        rangeMax = dataSet[:, i].max()  
        stepSize = (rangeMax - rangeMin) / numSteps  
        for j in range(m):                 
            threVal = rangeMin + float(j) * stepSize  
            for unequal in ['lt', 'gt']:    
                err = calErr(dataSet, i, threVal, unequal, D)  
                if err < bestErr:          
                    bestErr = err
                    bestStump["feature"] = i
                    bestStump["threshVal"] = threVal
                    bestStump["unequal"] = unequal
                    bestStump["err"] = err

    return bestStump

def predict(data, bestStump):
    if bestStump["unequal"] == 'lt':
        if data[bestStump["feature"]] <= bestStump["threshVal"]:
            return 1
        else:
            return -1
    else:
        if data[bestStump["feature"]] >= bestStump["threshVal"]:
            return 1
        else:
            return -1

def AdaBoost(dataSet, T):
    m, n = dataSet.shape
    D = np.ones((1, m)) / m                      
    classLabel = dataSet[:, -1].reshape(1, -1)   
    G = {}      

    for t in range(T):
        stump = buildStump(dataSet, D)           
        err = stump["err"]
        alpha = np.log((1 - err) / err) / 2      
        pre = np.zeros((1, m))
        for i in range(m):
            pre[0][i] = predict(dataSet[i], stump)
        a = np.exp(-alpha * classLabel * pre)
        D = D * a / np.dot(D, a.T)

        G[t] = {}
        G[t]["alpha"] = alpha
        G[t]["stump"] = stump
    return G

def adaPredic(data, G):
    score = 0
    for key in G.keys():
        pre = predict(data, G[key]["stump"])  
        score += G[key]["alpha"] * pre        
    flag = 0
    if score > 0:
        flag = 1
    else:
        flag = -1
    return flag

def calcAcc(dataSet, G):
    rightCnt = 0
    for data in dataSet:
        pre = adaPredic(data, G)
        if pre == data[-1]:
            rightCnt += 1
    return rightCnt / float(len(dataSet))

def plotData(data, clf):
    X1, X2 = [], []
    Y1, Y2 = [], []
    datas=data
    labels=data[:,2]
    for data, label in zip(datas, labels):
        if label > 0:
            X1.append(data[0])
            Y1.append(data[1])
        else:
            X2.append(data[0])
            Y2.append(data[1])

    x = linspace(0, 0.8, 100)
    y = linspace(0, 0.6, 100)

    for key in clf.keys():
        z = [clf[key]["stump"]["threshVal"]]*100
        if clf[key]["stump"]["feature"] == 0:
            plt.plot(z, y)
        else:
            plt.plot(x, z)

    plt.scatter(X1, Y1, marker='+', label='好瓜', color='b')
    plt.scatter(X2, Y2, marker='_', label='坏瓜', color='r')

    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.xlim(0, 0.8) 
    plt.ylim(0, 0.6)  
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.legend(loc='upper left')
    plt.show()

def main():
    dataSet = DataSet()
    for t in [3, 5, 11]:  
        G = AdaBoost(dataSet, t)
        print('准确率=',calcAcc(dataSet, G))
        plotData(dataSet,G)
        
if __name__ == '__main__':
    main()
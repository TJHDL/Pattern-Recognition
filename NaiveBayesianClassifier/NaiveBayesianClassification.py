import pandas as pd
import math
import numpy as np

def readfile():
    datasets = pd.read_csv(r'WaterMelon3.0.csv', encoding = 'gbk')
    del datasets['Idx']
    return datasets

def Continuous_ProbDensity_u(data):
    sum = 0.0
    for k in range(len(data)):
        sum += float(data[k])
    ave = sum / len(data)
    return ave

def Continuous_ProbDensity_sigma2(data,ave):
    sigma2 = 0.0
    for k in range(len(data)):
        sigma2 += pow((float(data[k]) - ave),2)
    sigma2 /= len(data)
    return sigma2

def Continuous_ProbDensity(data, ave, sigma2):
    P_PD = (1 / (math.sqrt(2*math.pi) * math.sqrt(sigma2))) * math.exp((-1) * pow((data - ave),2) / (2 * sigma2))
    return P_PD

def NaiveBayesianClassifier(test_data):
    train_data = readfile()
    train_data = train_data.values.tolist()
    
    pos_data = []
    neg_data = []
    for i in range(len(train_data)):
        if train_data[i][8] == 1:
            pos_data.append(train_data[i])
        else:
            neg_data.append(train_data[i])
    P_pos = (len(pos_data)+1) / (len(train_data)+2)
    P_neg = (len(pos_data)+1) / (len(train_data)+2)
    
    
    for i in range(len(test_data) - 1):
        if type(test_data[i]).__name__ != 'float' and type(test_data[i]).__name__ != 'int':
            x = 0
            Ni = 3
            for j in range(len(pos_data)):
                if test_data[i] == pos_data[j][i]:
                    x += 1
            if i == 5:
                Ni = 2
            P_pos *= (x + 1) / (len(pos_data) + Ni)
        else:
            pos_data = np.array(pos_data)
            ave = Continuous_ProbDensity_u(pos_data[:,i])
            sigma2 = Continuous_ProbDensity_sigma2(pos_data[:,i], ave)
            P_PD = Continuous_ProbDensity(test_data[i], ave , sigma2)
            P_pos *= P_PD
    
    for i in range(len(test_data) - 1):
        if type(test_data[i]).__name__ != 'float' and type(test_data[i]).__name__ != 'int':
            x = 0
            Ni = 3
            for j in range(len(neg_data)):
                if test_data[i] == neg_data[j][i]:
                    x += 1
            if i == 5:
                Ni = 2
            P_neg *= (x + 1) / (len(neg_data) + Ni)
        else:
            neg_data = np.array(neg_data)
            ave = Continuous_ProbDensity_u(neg_data[:,i])
            sigma2 = Continuous_ProbDensity_sigma2(neg_data[:,i], ave)
            P_PD = Continuous_ProbDensity(test_data[i], ave , sigma2)
            P_neg *= P_PD
            
    print("朴素贝叶斯分类器分类结果如下：")   
    print("测试例为好瓜的概率：", P_pos)
    print("测试例为坏瓜的概率：", P_neg)
    if P_pos >= P_neg:
        print("测试例是好瓜")
    else:
        print("测试例是坏瓜")

if __name__ == '__main__':
    test_data = ['dark_green','curl_up','little_heavily','distinct','sinking','hard_smooth',0.697,0.460]
    NaiveBayesianClassifier(test_data)
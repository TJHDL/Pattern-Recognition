import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt

#根据信息熵增益率确定最优划分属性
def cal_Ent(dataset):
    total_row = len(dataset)
    label_dict = {}
    for attribute_vector in dataset:
        label = attribute_vector[-1]
        if label not in label_dict.keys():
            label_dict[label] = 0
        label_dict[label] += 1
    Ent = 0.0
    for key in label_dict:
        p = float(label_dict[key]) / total_row
        Ent -= p * log(p,2) 
    return Ent

#根据离散属性划分数据集
def split_discrete_attribute(dataset,col,value):
    ret_dataset = []
    for row in dataset:
        if row[col] == value:
            ret_dataset_row = row[:col]
            ret_dataset_row.extend(row[col+1:])
            ret_dataset.append(ret_dataset_row)
    return ret_dataset

#根据连续属性划分数据集
def split_continuous_attribute(dataset,col,value,direction):
    ret_dataset = []
    for row in dataset:
        if direction == 0:
            if row[col] > value:
                ret_dataset_row = row[:col]
                ret_dataset_row.extend(row[col+1:])
                ret_dataset.append(ret_dataset_row)
        else:
            if row[col] <= value:
                ret_dataset_row = row[:col]
                ret_dataset_row.extend(row[col+1:])
                ret_dataset.append(ret_dataset_row)
    return ret_dataset

#选择最优划分属性
def choose_best_attribute(dataset,label):
    attribute_num = len(dataset[0]) - 1
    base_Ent = cal_Ent(dataset)
    Gain = 0.0
    Gain_ratio = 0.0
    best_attribute = -1
    best_split_dict = {}
    Gain_list = []
    IV_list = []
    Gain_ratio_list = []
    for i in range(attribute_num):
        attribute_list = [exm[i] for exm in dataset]
        if type(attribute_list[0]).__name__ == 'float' or type(attribute_list[0]).__name__ == 'int':
            sorted_attribute_list = sorted(attribute_list)
            Ta = []
            for j in range(len(sorted_attribute_list)-1):
                Ta.append((sorted_attribute_list[j]+sorted_attribute_list[j+1]) / 2.0)
            Ta_len = len(Ta)
            best_Ent = 10000
            IV = 0.0
            for j in range(Ta_len):
                new_Ent = 0.0
                sub_dataset0 = split_continuous_attribute(dataset,i,Ta[j],0)
                sub_dataset1 = split_continuous_attribute(dataset,i,Ta[j],1)
                p0 = float(len(sub_dataset0)) / len(dataset)
                p1 = float(len(sub_dataset1)) / len(dataset)
                new_Ent += p0 * cal_Ent(sub_dataset0)
                new_Ent += p1 * cal_Ent(sub_dataset1)
                if new_Ent < best_Ent:
                    best_Ent = new_Ent
                    best_split = j
            best_split_dict[label[i]] = Ta[best_split]
            Gain = base_Ent - best_Ent
            Gain_list.append(Gain)
            sub_dataset0 = split_continuous_attribute(dataset,i,Ta[best_split],0)
            sub_dataset1 = split_continuous_attribute(dataset,i,Ta[best_split],1)
            p0 = float(len(sub_dataset0)) / len(dataset)
            p1 = float(len(sub_dataset1)) / len(dataset)
            IV -= p0 * log(p0,2)
            IV -= p1 * log(p1,2)
            IV_list.append(IV)
            Gain_ratio = Gain / IV
            Gain_ratio_list.append(Gain_ratio)
            
        else:
            new_Ent = 0.0
            IV = 0.0
            discrete_value = set(attribute_list)
            for value in discrete_value:
                sub_dataset = split_discrete_attribute(dataset,i,value)
                p = float(len(sub_dataset)) / len(dataset)
                new_Ent += p * cal_Ent(sub_dataset)
                IV -= p * log(p,2)
            Gain = base_Ent - new_Ent
            Gain_list.append(Gain)
            IV_list.append(IV)
            Gain_ratio = Gain / IV
            Gain_ratio_list.append(Gain_ratio)
    Gain_sum = 0.0
    for i in range(len(Gain_list)):
        Gain_sum += Gain_list[i]
    Gain_ave = Gain_sum / len(Gain_list)
    highest_Gain_ratio = 0.0
    for i in range(len(Gain_list)):
        if Gain_list[i] >= Gain_ave:
            if Gain_ratio_list[i] >= highest_Gain_ratio:
                highest_Gain_ratio = Gain_ratio_list[i]
                best_attribute = i
    if type(attribute_list[0]).__name__ == 'float' or type(attribute_list[0]).__name__ == 'int':
        best_split_value = best_split_dict[label[best_attribute]]
        label[best_attribute] = label[best_attribute] + '<=' + str(best_split_value)
        for i in range(len(dataset[0])):
            if dataset[i][best_attribute] <= best_split_value:
                dataset[i][best_attribute] = 1
            else:
                dataset[i][best_attribute] = 0
    return best_attribute

#属性为空集或样本在属性上的取值相同但仍未分完
def Vote(D_list):
    num_count = {}
    for exm in D_list:
        if exm not in num_count.keys():
            num_count[exm] = 0
        num_count[exm] += 1
    return max(num_count)
            
#递归生成决策树
def create_tree(dataset,label):
    D_list = [example[-1] for example in dataset]
    if D_list.count(D_list[0]) == len(D_list):
        return D_list[0]
    pre_example = dataset[0]
    flag = 0
    for example in dataset:
        if example[1:-2] != pre_example[1:-2]:
            flag = 1
            break
        pre_example = example
    if len(dataset[0]) == 1 or flag == 0:
        return Vote(D_list)
    best_attribute = choose_best_attribute(dataset,label)
    best_attribute_label = label[best_attribute]
    My_Tree = {best_attribute_label:{}}
    del(label[best_attribute])
    attribute_value = [example[best_attribute] for example in dataset]
    value = set(attribute_value)
    for val in value:
        sublabel = label[:]
        My_Tree[best_attribute_label][val] = create_tree(split_discrete_attribute(dataset,best_attribute,val),sublabel)
    return My_Tree

#计算树的叶子节点数量
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else: numLeafs+=1
    return numLeafs
 
#计算树的最大深度
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth
 
#画节点
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
    xytext=centerPt,textcoords='axes fraction',va="center", ha="center",\
    bbox=nodeType,arrowprops=arrow_args)
 
#画箭头上的文字
def plotMidText(cntrPt,parentPt,txtString):
    lens=len(txtString)
    xMid=(parentPt[0]+cntrPt[0])/2.0-lens*0.002
    yMid=(parentPt[1]+cntrPt[1])/2.0
    createPlot.ax1.text(xMid,yMid,txtString)
    
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.x0ff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.y0ff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.y0ff=plotTree.y0ff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.x0ff=plotTree.x0ff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.x0ff,plotTree.y0ff),cntrPt,leafNode)
            plotMidText((plotTree.x0ff,plotTree.y0ff),cntrPt,str(key))
    plotTree.y0ff=plotTree.y0ff+1.0/plotTree.totalD
 
def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.x0ff=-0.5/plotTree.totalW
    plotTree.y0ff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
               
df=pd.read_csv('4.3_watermelon_data_3.0_C4.5.csv')
data=df.values[:,1:].tolist()
data_full=data[:]
labels=df.columns.values[1:-1].tolist()
labels_full=labels[:]
My_Tree=create_tree(data,labels)

decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

createPlot(My_Tree)
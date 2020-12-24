import numpy as np
import matplotlib.pyplot as plt
import math

def DataSet():
    dataSet = [
        [0.697, 0.460],
        [0.774, 0.376],
        [0.634, 0.264],
        [0.608, 0.318],
        [0.556, 0.215],
        [0.403, 0.237],
        [0.481, 0.149],
        [0.437, 0.211],
        [0.666, 0.091],
        [0.243, 0.267],
        [0.245, 0.057],
        [0.343, 0.099],
        [0.639, 0.161],
        [0.657, 0.198],
        [0.360, 0.370],
        [0.593, 0.042],
        [0.719, 0.103],
        [0.359, 0.188],
        [0.339, 0.241],
        [0.282, 0.257],
        [0.748, 0.232],
        [0.714, 0.346],
        [0.483, 0.312],
        [0.478, 0.437],
        [0.525, 0.369],
        [0.751, 0.489],
        [0.532, 0.472],
        [0.473, 0.376],
        [0.725, 0.445],
        [0.446, 0.459],
    ]

    labels = ['密度', '含糖率']

    return np.array(dataSet), labels

def k_means(dataSet, k):
    dataset = dataSet[:, :]
    if k == 3:
        #初始均值向量分布适中
        Ave_Vec_Index = [5, 11, 26] 
        #初始均值向量分布分散
        #Ave_Vec_Index = [9, 16, 28]
        #初始均值向量分布密集
        #Ave_Vec_Index = [12, 13, 2]
    elif k == 5:
        #初始均值向量分布适中
        Ave_Vec_Index = [5, 11, 26, 3, 20]
        #初始均值向量分布分散
        #Ave_Vec_Index = [9, 16, 28, 14, 24]
        #初始均值向量分布密集
        #Ave_Vec_Index = [12, 13, 2, 24, 3]
    
    Ave_Vec = dataset[Ave_Vec_Index]

    Cluster_List = {}
    while True:
        cluster = {}
        for i in range(len(dataSet)):
            Min_Dist = np.inf
            Min_Index = -1
            for j in range(len(Ave_Vec)):
                dist = math.sqrt((float(dataset[i][0]) - float(Ave_Vec[j][0]))**2 + (float(dataset[i][1]) - float(Ave_Vec[j][1]))**2)
                if dist < Min_Dist:
                    Min_Dist = dist
                    Min_Index = j
            
            if Min_Index not in cluster.keys():
                cluster[Min_Index] = []
            cluster[Min_Index].append(i)

        cnt = 0  
        if k == 3:
            New_Vec = [[0 for i in range(2)] for i in range(3)]
        elif k == 5:    
            New_Vec = [[0 for i in range(2)] for i in range(5)]
        
        for i in range(len(Ave_Vec)):
            data = np.array(dataset[cluster[i]])
            T_x = 0
            T_y = 0
            for j in range(len(data)):
                T_x += data[j][0]
                T_y += data[j][1]
            New_Vec[i][0] = T_x / len(data)  
            New_Vec[i][1] = T_y / len(data)
            Delta_Dist = math.sqrt((float(New_Vec[i][0]) - float(Ave_Vec[i][0]))**2 + (float(New_Vec[i][1]) - float(Ave_Vec[i][1]))**2)
            if Delta_Dist != 0:
                Ave_Vec[i] = New_Vec[i]
                cnt += 1

        if cnt == 0:
            Cluster_List = cluster
            break

    return Cluster_List, Ave_Vec

def main():
    dataSet, labels = DataSet()
    Cluster_List, Ave_Vec = k_means(dataSet, 3)
    #Cluster_List, Ave_Vec = k_means(dataSet, 5)
    print("Cluster result:", Cluster_List)
    print("Mean Vector result:", Ave_Vec)
    for key in Cluster_List.keys():
        data = np.array(dataSet[Cluster_List[key]])
        plt.scatter(data[:, 0], data[:, 1], label=key)
    plt.scatter(Ave_Vec[:, 0], Ave_Vec[:, 1], s=80, c='r', marker="+")
    plt.xlabel("密度")
    plt.ylabel("含糖率")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
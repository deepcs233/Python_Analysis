#encoding=utf-8

import random
import numpy
import pandas as pd

class Cluster(object):

    def __init__(self):
        self.points=[]
        self.centroid=None

    def addPoint(self,point):
        self.points.append(point)

    def setNewCentroid(self):

        
        if len(self.points)==0:
            print 'No neighbour'
            return self.centroid

        
        self.centroid=[]
        for i in range(len(self.points[0])):
            temp=[]
            for j in range(len(self.points)):
                temp.append(self.points[j][i])

            self.centroid.append(sum(temp)/len(temp))
        
        self.points=[]
        return self.centroid

class Kmeans(object):
    #bestIter is 20

    def __init__(self,k=3,maxIter=20,minDistan=1):

        self.k=k
        self.maxIter=maxIter
        self.minDistan=minDistan

    def run(self,data):
        self.data=numpy.array(data)
        self.clusters=[None]*self.k
        self.lastClusters=None
        randomData=random.sample(range(len(self.data)),self.k)
        

        #随机选取k组数作为初始中心
        for idx in range(self.k):
            self.clusters[idx]=Cluster()
            self.clusters[idx].centroid=self.data[randomData[idx]] 
            
        iterations = 0
        
        while self.shouldExit(iterations) is False:
            self.lastClusters=[cluster.centroid for cluster in self.clusters]
            

            for each in self.data:
                #无限大
                mindis=float('Inf')
                for cluster in self.clusters:
                    
                    dis=self.calcDistance(each,cluster.centroid)
                    if dis<mindis:
                        mindis=dis
                        nearest=cluster

                nearest.addPoint(each)
                        
            for cluster in self.clusters:
                 cluster.setNewCentroid()
               
            iterations += 1

        return [cluster.centroid for cluster in self.clusters]

    def shouldExit(self,iterations):
        
        if self.lastClusters==None:
            return False

        acceptPoint=0
        #计算新的中心和老的中心之间的距离
        dis=[]
        for idx in range(self.k):
            dist=self.calcDistance(
                numpy.array(self.clusters[idx].centroid),
                numpy.array(self.lastClusters[idx])
                )
            dis.append(dist)
            if dist<self.minDistan: acceptPoint+=1
            
        print 'It\'s '+str(iterations)+'  iteration'
        print 'Accept point:',str(acceptPoint)
        print 'MeanDistance:',str(sum(dis)/len(dis))
        print 'MaxDistance:',max(dis)
        print 'MinDistance:',min(dis)
        print '*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*'*2
        
        
        if acceptPoint==self.k:
            return True

        if iterations <=self.maxIter:
            return False

        return True

    def calcDistance(self,a,b):
        result=numpy.sqrt(sum((a-b)**2))
        return result


if __name__=='__main__':

    TRAIN_NUM=42000
    data=pd.read_csv('train.csv')
    train_data = data.values[0:TRAIN_NUM,1:]
    train_label = data.values[0:TRAIN_NUM,0]
    kk=Kmeans(k=10)
    res=kk.run(train_data)

    with open('numbers.txt','w') as f:
        f.write(str(res))

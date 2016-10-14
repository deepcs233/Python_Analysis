#encoding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class preception():

    def __init__(self,maxIter=100000,data=[],cla=[],step=1):

        self.data=data
        self.dimension=len(data[0])
        self.weight=np.zeros(self.dimension)
        self.bias=0
        self.step=step
        self.maxIter=maxIter
        self.cla=cla

        if len(data)!=len(cla):
            print '数据数与分类信息数不等'

    def step_(self,array,cla):

        if cla*(array.dot(self.weight)+self.bias) <=0:

            self.weight+=self.step*cla*array
            self.bias+=self.step*cla
            #print self.weight
            
            return 1


        else:
            return 0

    def run(self):
        data=np.array(self.data)
        result=0
        length=len(data)
        j=0
        for i in xrange(self.maxIter):

            while(self.step_(data[j],self.cla[j])):
                result=0

            result+=1
            j+=1
            #print result
            if j==length:
                if result>=length:

                    #如果是二维图像则画图
                    if self.dimension==2:
                        self.plot()
                    return (self.weight,self.bias)
                
                else:
                    j=0

        return 0

    def plot(self):
        
        #x,y为散点的横，纵坐标序列
        x=[]
        y=[]
        for each in self.data:
            x.append(each[0])
            y.append(each[1])

        #k,b 为直线参数,xx,yy为直线点参数序列
        k=-self.weight[0]/self.weight[1]
        b=-self.bias/self.weight[1]
        xx=np.arange(min(min(x),min(y))-2,max(max(x),max(y))+2,0.01)
        yy=k*xx+b
        
        fig = plt.figure(figsize=(8, 5), dpi=80)
        ax=fig.add_subplot(111)

        for i in xrange(len(self.cla)):
            if self.cla[i]>0:
                #绘制正分类点
                ax.scatter(x[i],y[i],c='red')
            else:
                #绘制负分类点
                ax.scatter(x[i],y[i],c='green')
            
        #绘制直线
        ax.plot(xx,yy)
        
        plt.show()
        
if __name__=='__main__':
    d=preception(data=[[3,3],[4,3],[1,1],[2,2],[5,7],[4,6]],cla=[1,1,-1,-1,-1,-1])
    res=d.run()

            

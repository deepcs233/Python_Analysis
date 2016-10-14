# -*- coding: utf-8 -*-
# CopyRight by heibanke

import pandas as pd
import numpy as np
import time
import csv
import random
from functools import wraps,partial
import copy


def geneticoptimize(domain,costf,popsize=10,step=1,mutprob=0.2,elite=0.2,maxIter=100):
    #变异
    def mutate(vec):

        i=random.randint(0,len(domain)-1)

        if random.random()<0.5 and vec[i]>domain[i][0]:
            return vec[0:i]+[vec[i]-step]+vec[i+1:]
        elif vec[i]<domain[i][1]:
            return vec[0:i]+[vec[i]+step]+vec[i+1:]
        return vec

    #交叉
    def crossover(r1,r2):
        res=[]
        for i in range(len(r1)):
            if random.random()<0.5:
                res.append(r1[i])
            else:
                res.append(r2[i])
        return res

    #构造初始种群
    pop=[]
    
    for i in range(popsize):
        vec=[random.choice(domain[i]) for i in range(len(domain))]
        pop.append(vec)
    print pop
    topelite=int(elite*popsize)

    #main
    for i in range(maxIter):
        scores=[(costf(*v),v) for v in pop]
        scores.sort()
        ranked=[v for (s,v)in scores]

        pop=ranked[0:topelite]

        while len(pop)<popsize:
            if random.random()<mutprob:
                c=random.randint(0,topelite)
                pop.append(mutate(ranked[c]))

            else:
                c1=random.randint(0,topelite)
                c2=random.randint(0,topelite)
                pop.append(crossover(ranked[c1],ranked[c2]))


    print scores[0][0]

    return scores[0][0],scores[0][1]
        
def time_cost(timef):#传递装饰函数的参数,此处的timef为运行次数
    def decorator(f):#传递装饰的目标函数
        @wraps(f)
        def _f(*args, **kwargs):#传递目标函数的参数
            best_time=10**10
            all_time=0
            worst_time=0
            for i in range(timef):
                start=time.clock()
                a=f(*args, **kwargs)
                end=time.clock()
                time_a=end-start
                if time_a<best_time:
                    best_time=time_a
                if time_a>worst_time:
                    worst_time=time_a
                all_time=all_time+time_a
            print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'*2
            print f.__name__,"the best of run cost time is",best_time
            print f.__name__,"the avg of run cost time is",all_time/timef
            print f.__name__,"the worst of run cost time is",worst_time
            print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'*2
            
            return a
        return _f
    return decorator



def normalize2(x):
    """
    linalg.norm(x), return sum(abs(xi)**2)**0.5
    apply_along_axis(func, axis, x),
    """
    norms = np.apply_along_axis(np.linalg.norm, 1, x) + 1.0e-7
    return x / np.expand_dims(norms, -1)


#k为纳入考虑的最近k个元素,inWeight指示分析最近点时是否考虑因距离而产生的权重加成  
def nearest_neighbor(train_x, train_y, test_x,k=1,inWeight=True):
    
    train_x = normalize2(train_x)
    
    test_x =  normalize2(test_x)
 
    print test_x.shape
    print train_x.shape
    corr = np.dot(test_x, np.transpose(train_x))
    print 'start sorting'
    '''
    argmax = np.argmax(corr, axis=1)
    preds = train_y[argmax]
    '''
    preds=[]
    for i in xrange(len(test_x)):
        if i %1000==0: print str(i),'------- ok'
        row=corr[i]
        row=sorted(row,reverse=True)[0:k]
        ss={}
        ax=[]

        if inWeight:
            '''
            此处的w与下面的第二个for循环均为调整权重，即距离越近所占权重越高
            '''
            w=2*k
            for each in row:
                for j in range(w):
                
                    ax.append(train_y[list(corr[i]).index(each)])

                w-=1
        else:
            for each in row:
                ax.append(train_y[list(corr[i]).index(each)])
                
        for each in ax:
                      
            if each not in ss:
                ss[each]=0
            else:
                ss[each]+=1

        ss=sorted(ss.iteritems(),key=lambda x :x[1],reverse=True)
        preds.append(ss[0][0])

    return preds


def validate(preds, test_y):
    count = len(preds)
    correct = (preds == test_y).sum()
    return float(correct) / count

#交叉验证，rate为交叉验证时测试数据所占比例
#将需要调试的参数前置
def crossVali(data,label,k=1,inWeight=True,multiple=2,func=nearest_neighbor,rate=0.05):
    print str(k),str(inWeight),str(multiple)
    train=[]
    test=[]
    train_label=[]
    test_label=[]
    data=addNewFeature(data,label,multiple=2)   
    for i in range(len(data)):
        if random.random()<rate:
            test.append(data[i])
            test_label.append(label[i])
        else:
            train.append(data[i])
            train_label.append(label[i])
    
    
    preds=func(train,train_label,test,k,inWeight)

    
    acc=validate(np.array(preds),np.array(test_label))
    print 'acc is ',str(acc)
    return (1-acc)*100        

#为图片识别添加其他特征

def addNewFeature(data,label,size=(28,28),multiple=1):
    '''
    fill为填充度，饱和度
    symmetryvert为上下对称性
    symmetryHor=[]为水平对称性
    '''
    fill=[]
    symmetryvert=[]
    symmetryHor=[]
    p=time.time()
    print 'Start adding Feature'
    for i in range(len(data)):
        non_zero=sum(data[i]>0)
        fill.append(non_zero*multiple)
        left=0
        right=0
        up=0
        down=0
        for j in range(len(data[0])):
            if data[i][j]>0:
                if (j%size[0])<size[0]/2:
                    left+=1
                else:
                    right+=1
                if j/size[0]<size[1]/2:
                    up+=1
                else:
                    down+=1
        symmetryvert.append(int(float(abs(up-down)*1000)/non_zero)*multiple)
        symmetryHor.append(int(float(abs(left-right)*1000)/non_zero)*multiple)
        #print str(label[i]),str(fill[i]),str(int(float(abs(up-down)*1000)/non_zero)),str(int(float(abs(left-right)*1000)/non_zero))

    data=np.c_[data,fill,symmetryvert,symmetryHor]
    print 'Finish adding Feature,using '+str(time.time()-p)+' s'
    return data

def findBestPara(train,label,func,rate,timef,args,poi,domain):
    '''
    寻找最好的参数组合,func为学习函数
    poi为需要更改的参数索引位置,domain为全参数定义域
    rate为交叉验证时测试数据所占比例
    timef为相同参数下某函数的运行次数
    
    '''
    args_best=args
    acc_best=0
    for i in range(len(poi)):
        '''
        开始改变第i个参数
        '''
        for j in range(len(domain[poi[i]])):
            '''
            选取定义域中第j个参数
            '''
            t=time.time()
            args_t=copy.copy(args)
            args_t[poi[i]]=domain[poi[i]][j]
            print'<<<<<<<<<<<<<<<'*4
            print 'When paraments is'+str(args_t)+':'
            
            acc=crossVali(train,label,func,rate,*args_t)
            
            print("Validation Accuracy: %f, %.2fs" % (acc, time.time() - t))
            
            if acc>acc_best:
                args_best=args_t
                acc_best=acc
                
    return (args_best,acc_best)
            
    
    
if __name__=='__main__':
    TRAIN_NUM = 42000
    TEST_NUM = 28000

    
    data = pd.read_csv('train.csv')
    train_data = data.values[0:TRAIN_NUM,1:]
    train_label = data.values[0:TRAIN_NUM,0]

    data = pd.read_csv('test.csv')
    
    test_data = data.values


##norm_funcs = [normalize2]
##train_data=addNewFeature(train_data,label=train_label,multiple=2)
##for norm_f in norm_funcs:
##    t = time.time()
##
##    preds = nearest_neighbor(train_data, train_label, test_data)
##    #acc = validate(preds, test_label)
##    #print("%s Validation Accuracy: %f, %.2fs" % (norm_f.__name__,acc, time.time() - t))
##        for k in [1,3,5,7,10]:
##            print str(k)+'------'*5
##            acc=crossVali(train_data,train_label,rate=0.05,k=k)
##            print("%s Validation Accuracy: %f, %.2fs" % (norm_f.__name__,acc, time.time() - t))
##            t=time.time()
##
##
##
##
##writer = csv.writer(file('sample.csv', 'wb'))
##writer.writerow(['ImageId','Label'])
##print len(preds)
##for i in range(len(preds)):
##    writer.writerow([str(i+1),str(preds[i])])
##writer.close()
##
    args=[1,True]
    poi=[0,1]
    domain=[[1,2,3,5],[0,1],[1,2,3,5,7]]
    costf=partial(crossVali,train_data,train_label)
    #findBestPara(train_data,train_label,nearest_neighbor,0.05,3,args,poi,domain)
    #c=geneticoptimize(domain,costf,popsize=5,maxIter=10)
    p=costf(5,1,3)

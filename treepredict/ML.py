#encoding=utf-8

import numpy as np
import pandas as pd
import random
import time

def time_cost(timef):#传递装饰函数的参数,此处的timef为运行次数
    def decorator(f):#传递装饰的目标函数
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


class BasicML(object):
    
    def __init__(self,data,lable,maxIter=10000,learnRate=0.001):
        self.data=np.array(data)
        self.lable=np.array(lable)
        self.maxIter=maxIter
        self.learnRate=learnRate
        pass

    
    def NormalData(self,shape,dtype=None):
        '''
        正规化数据
        '''
        if data.shape!=shape:
            if len(data)!=shape[0]*shape[1]:
                print 'Size Error'
                return
            
            data=data.reshape(shape)
        '''
        设置默认数据格式
        '''
        if dtype:
            if dtype=='float':dtype='float32'
            if dtype=='int':dtype='int32'

            if hasattr(np,dtype):
                data=data.astype(getattr(np,dtype))
            else:
                print '非法的数据格式'

        return data
        
    def train(self,func):
    '''
    func为训练函数
    '''
        for i in xrange(self.maxIter):

            return preds

        return 0
    
    def forecast(self,weight,test_data):
    '''
    提供权重及函数，返回预测结果
    '''
        pass
        
    def validate(self,preds,lable)
    '''
    验证准确率的函数
    '''
        count = len(preds)
        correct = (preds == lable).sum()
        return float(correct) / count


    def crossVaild(self,train,label,rate,func_preds):
    '''
    交叉验证函数,rate为用于测试的数据所占的比率
    '''
        train_data=[]
        test=[]
        train_label=[]
        test_label=[]

        for i in range(len(train)):
            if random.random()<rate:
                test.append(train[i])
                test_label.append(label[i])
            else:
                train_data.append(train[i])
                train_label.append(label[i])
        preds=func_preds(train_data,train_label,test)

        return validate(np.array(preds),np.array(test_label))

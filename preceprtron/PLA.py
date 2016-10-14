# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def sgn(x):
    # 取符号函数：返回一个元素值为1或-1的向量
    for i in xrange(len(x)):
        if x[i] < 0:
            x[i] = -1
        else:
            x[i] = 1
    return x
    
def train(X, y, maxiter, epsilon = 0.01):
    (nrow, ncol) = np.shape(X)  # 求得矩阵X的两个维度
    W = np.random.rand(ncol)    # 初始化权重向量
    bias = 0.0                  # 初始化偏差值
    err = 0.0
    for i in xrange(maxiter):   # 根据指定的迭代次数进行循环
        beta = np.zeros(ncol)   # 临时变量：记录对权重向量的修正结果
        b = 0.0                 # 临时变量：记录对偏差的修正结果
        s = 0.0                 # 对每轮迭代中的预测结果错误数量进行累加
        yhat = sgn(X.dot(W) + bias)  # 根据当前参数做出预测
        res = (yhat != y)       # 与训练集的标记进行比较
        err = np.mean(res)      # 求得当前模型输出的错误情况（百分比）
        if err < epsilon:       # 提前结束训练（达到预期准确度目标）
            break
        else:
            for k in xrange(len(res)):  # 逐一处理当前模型产生的错误
                if(res[k]):             # 仅处理出错的情况
                    beta = beta + y[k]*(X[k].transpose()) # 注意是累加
                    b = b + y[k]        # 同样对偏差进行累加
                    s += 1              # 统计本轮预测结果出错的样本个数
         
        W = W + beta/s          # 更新权重向量
        bias = bias + b/s       # 更新偏差值，至此一轮迭代结束
                
        print W, bias           # 可选：帮助我们观察训练的过程
    return (W, bias)


def gen_dataA(size, nfea):
    X, y = make_classification(n_samples = size, n_features = nfea, random_state=0)
    return X, y

def gen_dataB(size, nfea):
    X = np.random.randn(size*nfea).reshape(size, nfea) * 5
    noise = np.random.randn(size)
    y = np.zeros(size)
    for i in xrange(len(X)):
        if noise[i] > 0:
            X[i][1] = 3 * X[i][0] + 2 + 30*noise[i]  
            y[i] = 1
        else:
            X[i][1] = 3 * X[i][0] - 2 + 30*noise[i]  
            y[i] = -1
            
    return X, y


def show_mat(x):
    for i in xrange(len(x)):
        print x[i]
        
def draw(X, y, W, bias):
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Spectral)
    
    # 蓝色线条为训练得到的决策边界
    pred = [(-W[0] * x /W[1]) for x in X[:, 0]]
    plt.plot(X[:, 0], pred, 'b-', linewidth = 2)
    
    # 红色线条为基本事实
    facts = [(3 * x ) for x in X[:, 0]]
    plt.plot(X[:, 0], facts, 'r-', linewidth = 2)
    
    plt.show()
    
    
    
if __name__ == '__main__':
#     X, y = gen_dataA(200, 5);
    X, y = gen_dataB(300, 2);
    
    maxiter = 1000
    (W, bias) = train(X, y, maxiter)
    
    yhat = sgn(X.dot(W) + bias)
    print "in sample error = %d" %(np.sum(yhat != y))
    
    draw(X, y, W, bias)
    

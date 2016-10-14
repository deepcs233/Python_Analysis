#encoding=utf-8
import time
import math
import random


'''
定义域 domin
成本函数 costf
'''

def randomoptimize(domin,costf,maxIter=1000):
    best=99999999
    '''
    bestr is index of best
    '''
    bestr=None
    for i in range(maxIter):
        '''
        create a random solve
        '''
        r=[random.randint(domin[j][0],domin[j][0]) for j in range(len(domain))]

        cost=costf(r)

        if cost<best:
            best=cost
            bestr=r

    return r
        

def hillclimb(domain,costf):
    '''
    create a random solve
    '''
    sol=[random.randint(domin[j][0],domin[j][0]) for j in range(len(domain))]

    while 1:
        '''
        创建相邻解的列表 neighbors
        '''

        neighbors=[]
        for j in range(len(domain)):
            '''
            在每个方向相对偏离一些
            '''
            if sol[j]>domain[j][0]:
                neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])
            if sol[j]<domain[j][1]:
                neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])

        currentCost=costf(sol)
        best=currentCost
        for j in range(len(neighbors)):
            currentCost=costf(neighbors[j])
            if currentCost<best:
                best=currentCost
                sol=neighbours[j]

        if best==current:
            break

        return sol


def annealingoptimize(domain,costf,T=10000.0,cool=0.95,step=1):
    '''
    create a random solve
    '''
    vec=[random.randint(domin[j][0],domin[j][0]) for j in range(len(domain))]

    '''
    当余温尚存之时
    '''
    
    while T>0.1:

        '''
        choice a random index
        '''
        i=random.randint(0,len(domain)-1)
        '''
        choice a random direction to change
        '''

        direction=random.randint(-step,step)
        '''
        copy a new list which represent solve
        '''
        vecb=vec[:]
        vecb[i]+=dir
        if vecb[i]<domain[i][0]: vecb[i]=domain[i][0]
        elif vecb[i]>domain[i][1]: vecb[i]=domain[i][1]

        costBefore=costf(vec)
        costCurrent=costf(vecb)

        if (eb<ea or random.random()<pow(math.e ,-(eb-ea)/T)):
            vec=vecb

        T=T*cool

    return vec

'''
遗传算法
'''

def geneticoptimize(domain,costf,popsize=50,step=1,mutprob=0.2,elite=0.2,maxiter=100):
    #变异
    def geneticoptimize(vec):

        i=random.randint(0,len(domin)-1)

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
        vec=[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        pop.append(vec)

    topelite=int(elite*popsize)

    #main
    for i in range(maxiter):
        scores=[(costf(v),v) for v in pop]
        scores.sort()
        ranked=[v for (s,v)in socres]

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
        

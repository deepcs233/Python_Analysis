#encoding=utf-8
from math import sqrt

#基于距离的相似度评价
def sim_distance(prefs,person1,person2):

    #得到一个shared_items的列表
    si={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item]=1

    if len(si)==0 :
        return 0

    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) for
                        item in si])#prefs[person1] if item in prefs[person2]])

    return 1/(1+sqrt(sum_of_squares))


#皮尔逊相关系数
def sim_pearson(prefs,p1,p2):
    
    #得到一个shared_items的列表
    si={}

    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item]=1

    n=len(si)
    if n==0:
        return 0

    #对所有偏好求和
    sum1=sum([prefs[p1][it] for it in si])
    sum2=sum([prefs[p2][it] for it in si])

    #求平方和
    sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq=sum([pow(prefs[p2][it],2) for it in si])

    #求乘积之和
    pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])

    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0:
        return 0
    r=num/den

    #r即为相关系数
    return r


def topMatches(prefs,person,n=5,similarity=sim_pearson):
    scores=[(similarity(prefs,person,other),other) for other in prefs if other!=person]

    scores.sort()
    scores.reverse()
    return scores[0:n]

#转置
def transform(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})

            result[item][person]=prefs[person][item]

    return result

#构造物品比较数据集

def calculateSimilarItems(prefs,n=10,similarity=sim_distance):
    result={}

    itemPrefs=transform(prefs)
    c=0
    for item in itemPrefs:
        c=c+1
        if c%100==0:
            print "%d/ %d" %(c.len(itemPrefs))

        #寻找最为相似的物品

        scores=topMatches(itemPrefs,item,n=n,similarity=similarity)
        result[item]=scores

    return result

#获得推荐

def getRecommendedItems(prefs,itemMatch,user):
    userRatings=prefs[user]
    scores={}
    totalSim={}

    for item ,rating in userRatings:

        for similarity ,item2 in itemMatch:
            if item2 in userRatings:
                continue

            
            scores.setdefault(item2,0)
            scores[item2]=similarity*rating

            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
    

    rankings=[(score/totalSim[item],item) for item,score in scores.items()]
    rankings.sort()
    rankings.reverse()
    return rankings

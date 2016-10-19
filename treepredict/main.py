#encoding=utf-8

print 'importing...'

import numpy as np
import csv
import pandas as pd
import treepredict as tp
import random

print 'training...'

data=pd.read_csv('train.csv')
data=data.reindex(columns=[ u'Pclass',u'Age', u'Sex', u'SibSp', u'Parch',
       u'Fare', u'Embarked',u'Survived'])

#data=data.fillna({'Age':data['Age'].median()})
values=data.values[:]
tree=tp.buildTree(values,threshold=0.0,scoref=tp.giniRate)
#tp.prune(tree)

print 'predicting...'
print 'the size of the tree:',tp.size(tree)
testdata=pd.read_csv('test.csv')
testdata=testdata.reindex(columns=[ u'Pclass',u'Age',u'Sex', u'SibSp', u'Parch',
       u'Fare', u'Embarked'])

#testdata=testdata.fillna({'Age':data['Age'].median()})
testValues=testdata.values[:]

results=[]

for each in testValues:
    counts=tp.classify(tree,each)
    res=sorted(counts.iteritems(),key=lambda x:x[1],reverse=True)[0][0]
    results.append(res)


##with open('result.csv','wb') as f:
##    writer=csv.writer(f)
##    writer.writerow(['PassengerId','Survived'])
##    for i in xrange(len(results)):
##        writer.writerow([str(i+892),str(results[i])])
##print 'drawing picture...'
##tp.drawtree(tree)

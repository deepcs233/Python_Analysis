#encoding=utf-8
from PIL import Image,ImageDraw
class decisionNode(object):
    '''
    决策树结点
    col -- 有待检验的判断条件所对应的列索引值
    value -- 使每条数据分类为True所对应的值
    results -- 除叶节点的分支结点此值为None
    tNode -- 分类为ture所转进的树
    '''
    def __init__(self,col=-1,value=None,tNode=None,fNode=None,result=None):
        self.col=col
        self.value=value
        self.result=result
        self.tNode=tNode
        self.fNode=fNode

def dividedest(rows,column,value):
    
    '''
    将数据集根据column位的value拆分成两部分
    '''
    rows_t=[]
    rows_f=[]
    for row in rows:
        if isinstance(row[column],int) or isinstance(row[column],float):
            if row[column]>value:
                rows_t.append(row)
            else:
                rows_f.append(row)

        else:
            if row[column]==value:
                rows_t.append(row)
            else:
                rows_f.append(row)
    return (rows_t,rows_f)
def countResult(rows):
    '''
    用于统计数据集中的结果分布情况，默认结果位于最后一列
    
    '''
    counts={}
    for row in rows:
        
        #print row
        try:
            result=row[-1]
        except:
            result=row
            
        #print result
        counts[result]=counts.setdefault(result,1)+1
    #print counts
    return counts
        
def giniimpurity(rows):
    '''
    基尼不纯度
    '''
    total=len(rows)
    counts=countResult(rows)
    imp=0.0
    for k1 in counts:
        p1=float(counts[k1])/total
        for k2 in counts:
            if k1==k2: continue
            p2=float(counts[k2])/total
            imp+=p1*p2

    return imp

def entropy(rows):
    '''
    熵
    '''
    
    from math import log
    total=len(rows)
    result=0.0
    counts_value=countResult(rows)
    #print counts_value
    for each in counts_value:
        p=float(counts_value[each])/total
        result+=p*log(p,2)

    result=-result
    return result

def buildTree(rows,scoref=entropy,threshold=0):
    '''
    递归构造树,scoref评分函数须提示分数越低越好
    '''
    if len(rows)==0:
        return decisionNode()
    current_score=scoref(rows)
    total=len(rows)

    best_gain=0.0
    best_criteria=None
    best_row=None
    
    for col in range(len(rows[0])-1):
        elementsInrol=set()
        for row in rows:
            elementsInrol.add(row[col])

        for value in elementsInrol:
            row1,row2=dividedest(rows,col,value)

            p1=len(row1)/total
            mutualInfo=p1*scoref(row1)+(1-p1)*scoref(row2)
            '''
            gain即为信息增益
            mutualInfo为互信息
            '''
            gain=current_score-mutualInfo
            if gain>best_gain and len(row1)>0 and len(row2)>0:

                best_gain=gain
                best_criteria=(col,value)
                best_row=(row1,row2)
    
    if best_gain>threshold:
        trueNode=buildTree(best_row[0])
        falseNode=buildTree(best_row[1])

        return decisionNode(col=best_criteria[0],value=best_criteria[1],tNode=trueNode,fNode=falseNode)
        
    else:
        counts=countResult(rows)
        
        return decisionNode(result=counts)
            
            
        
def classify(tree,test):
    '''
    输入测试数据及树，输出分类结果
    '''
    res=None
    if tree.result!=None:
        #print tree.result
        return tree.result
    else:
        if isinstance(tree.value,int) or isinstance(tree.value,float):
            if test[tree.col]>tree.value:
                res=classify(tree.tNode,test)
            else:
                res=classify(tree.fNode,test)
        else:
            if  test[tree.col]==tree.value:
                res=classify(tree.tNode,test)
            else:
                res=classify(tree.fNode,test)
    return res

def prune(tree,threshold=0,scoref=entropy):
    '''
    剪枝优化
    目前只处理左右子树均为叶子结点的剪枝情形
    '''
    if tree.tNode!=None:
        if tree.tNode.result==None:
            prune(tree.tNode,threshold)
    if tree.fNode!=None:
        if tree.fNode.result==None:
            prune(tree.fNode,threshold)
    
    if tree.tNode!=None and tree.fNode!=None:
        if tree.tNode.result!=None and tree.fNode.result!=None:
            tb=[]
            fb=[]
            for each in tree.tNode.result:
                tb+=[each]*tree.tNode.result[each]
            for each in tree.fNode.result:
                fb+=[each]*tree.fNode.result[each]
            gain=scoref(tb+fb)-0.5*(scoref(tb)+scoref(fb))

            if gain>threshold:
                tree.result=countResult(tb+fb)
                tree.tNode=None
                tree.fNode=None
            
def printTree(tree,indent=''):
    '''
    打印树
    '''
    if tree.result!=None:
        
        result=sorted(tree.result.iteritems(),key=lambda x:x[1],reverse=True)[0][0]
        print indent+'Result:'+result+'\n'

    else:
        print indent+str(tree.col)+':'+str(tree.value)+'?'+'\n'
        if tree.tNode!=None:
            print indent+'T:---->\n'
            printTree(tree.tNode,indent=indent+'\t')
        if tree.fNode!=None:
            print indent+'F:---->\n'
            printTree(tree.fNode,indent=indent+'\t')

        
        
def getwidth(tree):
    if tree.tNode==None and tree.fNode==None :return 1

    else:
        return getwidth(tree.tNode)+getwidth(tree.fNode)

def getdeepth(tree):
    if tree.tNode==None and tree.fNode==None :return 0

    else:
        return max(getdeepth(tree.tNode),getdeepth(tree.fNode))+1

def drawtree(tree,jpeg='tree.jpg'):
    w=getwidth(tree)*100
    h=getwidth(tree)*100+120
    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)

    drawnode(draw,tree,w/2,20)
    img.save(jpeg,'JPEG')

def drawnode(draw,tree,x,y):
    if tree.result==None:
        w1=getwidth(tree.fNode)*100
        w2=getwidth(tree.tNode)*100

        left=x-(w1+w2)/2
        right=x+(w1+w2)/2

        draw.text((x-20,y-10),str(tree.col)+':'+str(tree.value),(0,0,0))

        draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))
        draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))

        drawnode(draw,tree.fNode,left+w1/2,y+100)
        drawnode(draw,tree.tNode,right-w2/2,y+100)

    else:
        txt=' \n'.join(['%s:%d'%v for v in tree.result.items()])
        draw.text((x-20,y),txt,(0,0,0))

@property
def size(tree):
    return(getwidth(tree),getdeepth(tree))

import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score
import scipy.stats as stats




def pearsonCurr(x,y):
    #x,y are log(xi) and log(xj)
    #(scipy.stats.pearsonr(x,y)) return two numbers! r- pearsoncoeff, p- how much can we relay on the result
    b=scipy.stats.pearsonr(x,y)
    return (abs(b[0]),abs(b[1] ))

def top(allFtr, number=20):
    #:return: list of tuples (i,j)
    row=[]
    best=[]
    raw,col=allFtr.shape
    for i in range(col):
        for j in range(i,col):
            if i==j:
                continue
            r,p=pearsonCurr(allFtr[:,i],allFtr[:,j])
            row.append([r,i,j])
    rowSort=(sorted(row))
    if (number>len(rowSort)):
        #for the case where we are looking for more pairs of features (not likely)
        end=len(rowSort)
        print ("ERROR- number is bigger then number of features (from: top func)")
    else:
        end=  len(rowSort)-number
        for k in range(len(rowSort)-1,end-1,-1):
            pair=rowSort[k]
            best.append((pair[1],pair[2]))
    return best

#SCATTER PLOT DESIGN FUNCTION
def regrPlot(x_arr,y_arr,c,r,reg_y,p_value,title,x,y,path):
    plt.scatter(x_arr,y_arr,color='mediumaquamarine',marker="d",s=10)
    plt.plot(x_arr,reg_y, color='salmon', linewidth=1.5)
    patchC = mpatches.Patch(color='salmon', label='y = '+str(round(c,4))+'x + e')
    patchR = mpatches.Patch(color='salmon', label='r^2 = ' + str(round(r,4)))
    patchP = mpatches.Patch(color='salmon', label='p = ' + str(p_value))
    plt.legend(handles=[patchC ,patchR,patchP],fontsize = 'small',loc=2)
    plt.title(title)
    plt.xlabel(x,fontsize=10)
    plt.ylabel(y,fontsize=10)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()

#REGRESSION
def regFunc(x_np,y_np):
    x_list=x_np.tolist()
    y_list=y_np.tolist()
    #flat_list_x = [item for sublist in x_list for item in sublist]
    #flat_list_y = [item for sublist in y_list for item in sublist]
    #x=flat_list_x
    #y=flat_list_y
    x=x_list
    y=y_list
    regr = linear_model.LinearRegression()
    regr.fit(np.transpose(np.matrix(x)), np.transpose(np.matrix(y)))
    reg_y = regr.predict(np.transpose(np.matrix(x)))
    #FOR P- VALUE
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2 = r2_score(y, reg_y)
    coe = regr.coef_[0][0]
    result=(reg_y,r2,coe,p_value)
    return result

def CanculateCBvec(allFtr,graphFtr, ftrIndex,singleCoe=True):
    """
    :param allFtr: for the cij which is over all
    :param graphFtr: defined as ftrDict[graph_name]
    :param ftrIndex: list of tuples of best pair of features- [(1,2), (3,2), (i,j)...]
    :return: returns a vector b. bijk- a scalar for every pair of features(i,j) and graph k
    **This function is for a spesific graph!
    """
    bk=[]
    raw,col=allFtr.shape
    for i,j in ftrIndex:
        if singleCoe:
            reg_y,r2,coeij,p_value=regFunc(allFtr[:,i].T,allFtr[:,j].T)
        else:
            reg_y,r2,coeij, p_value=regFunc(graphFtr[:,i].T,graphFtr[:,j].T)
        xj=graphFtr[:,j]
        xi=graphFtr[:,i]
        bijkvec=xj-coeij*xi
        bkij=np.mean(bijkvec)
        bk.append(bkij)
    return np.asarray(bk)



#begin here:

def main(ftrDict, numPairFtrWanted=20):
    #initialize:
    numGraphs=len(ftrDict)
    for graph in ftrDict.values():
        vertexNum, ftrNum=graph.shape
        continue

    index=0

    allFtr=np.zeros((numGraphs*vertexNum,ftrNum))
    for graphFtr in ftrDict.values():
        allFtr[(index*vertexNum):((index+1)*vertexNum), :]=graphFtr
        index+=1


    # MKE SURE BK SRE STACKES AS BK IN A ROW
    currRow=0

    b=np.zeros((numGraphs,numPairFtrWanted))
    for graphFtrk in ftrDict.values():
        bk=CanculateCBvec(allFtr,graphFtrk, top(allFtr, numPairFtrWanted))
        b[currRow,:]= bk #maybe np.stack!
        currRow+=1

    return b


#tests:
"""
regrPlot(x,y,c,r,reg_y,p_value,"hey","x","y","pic")
graph1Ftr=np.matrix([[0,1,2],[1,3,5],[300,500,700]]).T
    graph2Ftr=np.matrix([[3, 4, 5], [7, 9, 11], [0,0,0]]).T
    ftrDict={}
    ftrDict['graph1Ftr']=graph1Ftr
    ftrDict['graph2Ftr']=graph2Ftr #all in log
    """



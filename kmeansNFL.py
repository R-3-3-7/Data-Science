# -*- coding: utf-8 -*-

import pylab, random
from KMEANS import *

#set line width
pylab.rcParams['lines.linewidth'] = 4
#set font size for titles 
pylab.rcParams['axes.titlesize'] = 20
#set font size for labels on axes
pylab.rcParams['axes.labelsize'] = 20
#set size of numbers on x-axis
pylab.rcParams['xtick.labelsize'] = 16
#set size of numbers on y-axis
pylab.rcParams['ytick.labelsize'] = 16
#set size of ticks on x-axis
pylab.rcParams['xtick.major.size'] = 7
#set size of ticks on y-axis
pylab.rcParams['ytick.major.size'] = 7
#set size of markers, e.g., circles representing points
#set numpoints for legend
pylab.rcParams['legend.numpoints'] = 1




def feature(featureDatafile):
    infile=open(featureDatafile,'r')
    line=infile.readline()
    feature={}
    ID=1
    for line in infile:
        feature[ID]=line.replace('\n','').split(',')
        ID+=1
    infile.close()
    return feature

def makeExamples(dataFile):
    f=feature(dataFile)
    examples=[]
    for key in f:
        examples.append(Example(str(key),[float(x) for x in f[key][:-1]],f[key][-1]))
    return examples
        
  
def plotClusters(clusterList,nc,n):
    pylab.rcParams['lines.linewidth']=2
    pylab.figure(n)
    pylab.title('NFL Draft')
    pylab.xlabel('Speed')
    pylab.ylabel('Agility')
    colors=['ro','bo','go','ko','yo']
    k=0
    while k<=nc-1:
        for example in clusterList[k].examples:
             pylab.plot(example.features[0],example.features[1],colors[k])
        k=k+1
            
ex = makeExamples('NFLDraft.txt') 
noCluster=Cluster(ex[:11])
yesCluster=Cluster(ex[11:])
print(ex)
plotClusters([noCluster,yesCluster],2,1)
c1=kmeans(ex,2)
plotClusters(c1,2,2)
         
    
    
    
    
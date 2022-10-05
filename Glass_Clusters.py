# -*- coding: utf-8 -*-
"""
CS235

Exam #2
"""

#1. Find six clusters using try kmeans on glassData.txt

from KMEANS import *
#Get my examples
infile=open('glassData.txt','r')
glassExam=[]
count=0
for line in infile:
    linelist=line.split(',')
    glassExam.append(Example(str(count),[float(x) for x in linelist[:-1]],label=linelist[-1][:1]))
    count+=1
infile.close()

#Get my clusters
glassClust=trykmeans(glassExam,6,200)


#2. Count and print the labels of each cluster without displaying the centroid
count=1
for clust in glassClust:
    total=0
    total = len(clust.examples)
    print('Cluster #{0} contains {1} items:'.format(count,total))
    for key in countCluster(clust):
        print('{0} items of glass type {0}'.format(countCluster(clust)[key],key))
    count+=1
    print('')
    
#3. A. Write a function called cohesion

def cohesion(cluster):
    """ a function that
computes the  reciprocal of the sum of the
distances from each point in the cluster to the centroid of the cluster of a cluster"""
    centroid=cluster.centroid.features
    dist = 0
    for exam in cluster.examples:
        dist+=minkowskiDist(exam.features,centroid,1)
    return 1/dist

#B. Write a function called seperation

def seperation(clust1,clust2):
    """ A function that computes the distance between the centroids of two clusters."""
    return minkowskiDist(clust1,clust2,1)

#Calculate the mean separation between clusters in the clustering and identify the two clusters with the greatest separation

from MaxClique import genPowerset    

clusters=genPowerset([0,1,2,3,4,5])[2]
greatestDist=[0,[]]
totalDist=0
dist=0
    
for i in range(len(clusters)):
    v1=clusters[i][0]
    v2=clusters[i][1]
    dist = minkowskiDist(glassClust[v1].centroid.features,glassClust[v2].centroid.features,1)
    if dist > greatestDist[0]:
        greatestDist[0]=dist
        greatestDist[1]=clusters[i]
    totalDist+=dist
    
mean=totalDist/len(clusters)

#C. Integrity of a cluster

def integrity(cluster):
    """Returns the integrity of a cluster"""
    total = 0
    most = 0
    count = countCluster(cluster)
    
    for key in count:
        if count[key] > most:
            most=count[key]
        total += count[key]
    return most/total
    
#This monster of a print statement
print('Cluster      1         2         3         4         5           6          ','-'*73,'Cohesion',' '*2,'{0}'.format(round(cohesion(glassClust[0]),4)),' '*2,'{0}'.format(round(cohesion(glassClust[1]),4)),' '*2,'{0}'.format(round(cohesion(glassClust[2]),4)),' '*2,'{0}'.format(round(cohesion(glassClust[3]),4)),' '*2,'{0}'.format(round(cohesion(glassClust[4]),4)),' '*2,'{0}'.format(round(cohesion(glassClust[5]),4)),'-'*73,'Integrity',' '*3,'{0}'.format(integrity(glassClust[0])),' '*3,'{0}'.format(integrity(glassClust[1])),' '*6,'{0}'.format(integrity(glassClust[2])),' '*2,'{0}'.format(round(integrity(glassClust[3]),4)),' '*2,'{0}'.format(integrity(glassClust[4])),' '*2,'{0}'.format(round(integrity(glassClust[5]),4)),'-'*73,'\n')

print('Global Integrity: {0}'.format(round((integrity(glassClust[0])+integrity(glassClust[1])+integrity(glassClust[2])+integrity(glassClust[3])+integrity(glassClust[4])+integrity(glassClust[5]))/6,6)))
print('Mean Seperation:{0}'.format(round(mean,6)))


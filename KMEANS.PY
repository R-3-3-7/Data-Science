import random,pylab

from clusterClass import *

def trykmeans(examples, numClusters, numTrials, verbose = False):
    """Calls kmeans numTrials times and returns the result with the
          lowest dissimilarity"""
    best = kmeans(examples, numClusters, verbose)
    minDissimilarity = dissimilarity(best)
    trial = 1
    while trial < numTrials:
        try:
            clusters = kmeans(examples, numClusters, verbose)
        except ValueError:
            continue #If failed, try again
        currDissimilarity = dissimilarity(clusters)
        if currDissimilarity < minDissimilarity:
            best = clusters
            minDissimilarity = currDissimilarity
        trial += 1
    return best

#figure 23.5
def kmeans(examples, k, verbose = False):
    #Get k randomly chosen initial centroids, create cluster for each
    initialCentroids = random.sample(examples, k)
    clusters = []
    for e in initialCentroids:
        clusters.append(Cluster([e]))
        
    #Iterate until centroids do not change
    converged = False
    numIterations = 0
    while not converged:
        numIterations += 1
        #Create a list containing k distinct empty lists
        newClusters = []
        for i in range(k):
            newClusters.append([])
            
        #Associate each example with closest centroid
        for e in examples:
            #Find the centroid closest to e
            smallestDistance = e.distance(clusters[0].getCentroid())
            index = 0
            for i in range(1, k):
                distance = e.distance(clusters[i].getCentroid())
                if distance < smallestDistance:
                    smallestDistance = distance
                    index = i
            #Add e to the list of examples for appropriate cluster
            newClusters[index].append(e)
            
        for c in newClusters: #Avoid having empty clusters
            if len(c) == 0:
                raise ValueError('Empty Cluster')
        
        #Update each cluster; check if a centroid has changed
        converged = True
        for i in range(k):
            if clusters[i].update(newClusters[i]) > 0.0:  
                #update returns distance between old and new centroids
                converged = False
                
        if verbose:
            print('Iteration #' + str(numIterations))
            for c in clusters:
                print(c)
            print('') #add blank line
    return clusters


def feature(featureDatafile):
    infile=open(featureDatafile,'r')
   
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
        examples.append(Example(str(key),[float(x) for x in f[key][:-1]]))
    return examples
        
def countCluster(cluster):
    """Returns the count of each label in the cluster."""
    label={}
    for exam in cluster.examples:
        if exam.label not in label.keys():
            label[exam.label]=1
        else:
            label[exam.label]+=1
    return label
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
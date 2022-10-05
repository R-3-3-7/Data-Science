import pylab

from ExampleClass import *

class Cluster(object):
    
    def __init__(self, examples):
        """Assumes examples a non-empty list of Examples"""
        self.examples = examples
        self.centroid = self.computeCentroid()
        
    def update(self, examples):
        """Assume examples is a non-empty list of Examples
           Replace examples; return amount centroid has changed"""
        oldCentroid = self.centroid
        self.examples = examples
        self.centroid = self.computeCentroid()
        return oldCentroid.distance(self.centroid)
    
    def computeCentroid(self):
        vals = pylab.array([0.0]*self.examples[0].dimensionality())
        for e in self.examples: #compute mean
            vals += e.getFeatures()
        centroid = Example('centroid', vals/len(self.examples))
        return centroid

    def getCentroid(self):
        return self.centroid

    def variability(self):
        totDist = 0.0
        for e in self.examples:
            totDist += (e.distance(self.centroid))**2
        return totDist
        
    
    def __str__(self,showCentroid = False):
        names = []
        for e in self.examples:
            names.append(e.getName())
        names.sort()
        if showCentroid :
            result = 'Cluster with centroid '\
               + str(self.centroid.getFeatures()) + ' contains:\n  '
            for e in names:
                   result = result + e + ', '
            return result[:-2] #remove trailing comma and space
        

def dissimilarity(clusters):
    """ Assumes clusters is a list of Cluster objects"""
    totDist = 0.0
    for c in clusters:
        totDist += c.variability()
    return totDist
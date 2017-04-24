'''
Created on Apr 22, 2017

@author: mohame11
'''
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from PIL import Image

def PCA(D, dim):
    '''
    #mean normalization
    normalizedD = np.zeros(D.shape)
    rows, cols = D.shape
    colsMeans = (D.sum(axis=0)) / float(cols)
    for i in range(rows):
        
    #cal covariance
    cov = np.cov(normalizedD.T)
    
    #eigen decomposintion
    e_vals, e_vecs = linalg.eig(cov)
    
    #reduce dim for the data
    for inst in dataSet:
        reducedFeatures = []
        for i in range(dim):
            reducedFeatures.append(np.dot(inst.featureVector, e_vecs[i].T))
        inst.featureVector = np.array(reducedFeatures)
        print inst.featureVector
    '''
    ###########
    normalizedD = D - D.mean(axis=0)
    cov = np.cov(normalizedD.T)
    e_vals, e_vecs = linalg.eig(cov)
    idx = np.argsort(e_vals)[::-1]
    e_vecs = e_vecs[:,idx]
    # sort eigenvectors according to same index
    evals = e_vals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    #selected_e_vecs = e_vecs[:, :dim]
    transformedD = np.dot(e_vecs.T, normalizedD.T).T[:,:dim]
    
    return transformedD, e_vecs, e_vals


def plotEvecs(vecs, dim, l, w):
    for r in range(dim):
        v = np.array(vecs[r], 'd')
        im = v.reshape(l,w)
        plt.imshow(im, cmap='gray')
        plt.savefig('evec'+str(r)+'.pdf', bbox_inches='tight')
        #plt.show()
        
    #img = 
   
    


def visualizeClusters(clusters, title, dataSet):
    colorMap = {'c0':'blue', 'c1':'green', 'c2':'red', 'c3':'cyan', 'c4':'magenta', 'c5':'yellow', 'c6':'black', 'c7':'grey', 'c8':'orange', 'c9':'pink'}
    X = []
    Y = []
    colors = []
    for i,c in enumerate(clusters):
        color = colorMap['c'+str(i)]
        for m in c.members:
            X.append(dataSet[m].featureVector[0])
            Y.append(dataSet[m].featureVector[1])
            colors.append(color)
    
    plt.scatter(X, Y, s=80, c=colors,label=str(colorMap))
    plt.legend(loc='upper center', shadow=True, prop={'size':10})
    plt.savefig(title+'.pdf', bbox_inches='tight')
    plt.show()
    
def visualizeClustersClasses(clusters, title, dataSet):
    plt.Figure()
    colorMap = {'0':'blue', '1':'green', '2':'red', '3':'cyan', '4':'magenta', '5':'yellow', '6':'black', '7':'grey', '8':'orange', '9':'pink'}
    X = []
    Y = []
    colors = []
    for i,c in enumerate(clusters):
        for m in c.members:
            X.append(dataSet[m].featureVector[0])
            Y.append(dataSet[m].featureVector[1])
            colors.append(colorMap[dataSet[m].classLabel])
    
    plt.scatter(X, Y, s=80, c=colors,label=str(colorMap))
    plt.legend(loc='upper center', shadow=True, prop={'size':10})
    plt.savefig(title+'.pdf', bbox_inches='tight')
    plt.show()
       


def visualizeSamples(samples, title):
    colorMap = {'0':'blue', '1':'green', '2':'red', '3':'cyan', '4':'magenta', '5':'yellow', '6':'black', '7':'grey', '8':'orange', '9':'pink'}
    
    X = []
    Y = []
    colors = []
    for i in samples:
        X.append(i.featureVector[0])
        Y.append(i.featureVector[1])
        colors.append(colorMap[i.classLabel])
    plt.scatter(X, Y, s=80, c=colors,label=str(colorMap))
    #plt.legend(bbox_to_anchor=(-0.1, 1.00, 1.00, .101), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size':10}) #legend font size
    plt.legend(loc='upper center', shadow=True, prop={'size':10})
    plt.savefig(title+'.pdf', bbox_inches='tight')
    plt.show()
    
    '''
    for i in samples:
        plt.scatter(i.featureVector[0], i.featureVector[1], c=colorMap[i.classLabel], marker="+", label=str(i.classLabel))
    
    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
    
    plt.savefig(title+'.pdf', bbox_inches='tight')
    plt.show()
    '''
    
    
    
    
    
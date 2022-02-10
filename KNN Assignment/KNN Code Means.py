import numpy as np
import matplotlib.pyplot as plt

#Read in data files
fileObj = open("animals.txt", "r")
animalData = fileObj.readlines()
fileObj.close()

fileObj = open("countries.txt", "r")
countryData = fileObj.readlines()
fileObj.close()

fileObj = open("fruits.txt", "r")
fruitData = fileObj.readlines()
fileObj.close()

fileObj = open("veggies.txt", "r")
veggieData = fileObj.readlines()
fileObj.close()

#Create cluster Arrays
#Cluster 1 - Animals
clust1 = np.zeros((len(animalData), len(animalData[0].split(' '))-1));

for iteration in range(len(animalData)):
    row = animalData[iteration].split(' ')
    clust1[iteration] = row[1:]
    
#Cluster 2 - Countries
clust2 = np.zeros((len(countryData), len(countryData[0].split(' '))-1));

for iteration in range(len(countryData)):
    row = countryData[iteration].split(' ')
    clust2[iteration] = row[1:]
    
#Cluster 3 - Fruit
clust3 = np.zeros((len(fruitData), len(fruitData[0].split(' '))-1));

for iteration in range(len(fruitData)):
    row = fruitData[iteration].split(' ')
    clust3[iteration] = row[1:]
    
#Cluster 4 - Veggies
clust4 = np.zeros((len(veggieData), len(veggieData[0].split(' '))-1));

for iteration in range(len(veggieData)):
    row = veggieData[iteration].split(' ')
    clust4[iteration] = row[1:]

#Create full Data Set
data = np.concatenate((clust1, clust2, clust3, clust4))
clusters = np.zeros((len(data),2))
currentCluster = 0


for i in range(len(data)):
    if(currentCluster == 0):
        clusters[i, 1] = 0
        if(i == len(clust1)):
            currentCluster += 1
    if(currentCluster == 1):
        clusters[i, 1] = 1
        if(i == (len(clust1) + len(clust2))):
            currentCluster += 1
    if(currentCluster == 2):
        clusters[i, 1] = 2
        if(i == (len(clust1) + len(clust2)+ len(clust3))):
            currentCluster += 1
    if(currentCluster == 3):
        clusters[i, 1] = 3
    

#Plotting the clusters
def cluster_plots(dataset, medoidInd=[], colours = 'gray' , title = 'Dataset'):
    fig,ax = plt.subplots()
    fig.set_size_inches(5, 5)
    ax.set_title(title,fontsize=14)
    ax.set_xlabel("Feature 0 Value",fontsize=14)
    ax.set_ylabel("Feature 1 Value",fontsize=14)
    ax.set_xlim(min(dataset[:,0]), max(dataset[:,0]))
    ax.set_ylim(min(dataset[:,1]), max(dataset[:,1]))
    
    #plotting cluster by colour
    colours = 'red', 'blue', 'green', 'orange', 'purple', 'gold', 'gray', 'pink', 'cyan'
    
    
    for i in range(len(dataset)):
        colour = colours[0]
        if(len(medoidInd) > 0):
            colourIndex = np.where(medoidInd == clusters[i,0])
            colour = colours[colourIndex[0][0]]
        ax.scatter(dataset[i, 0], dataset[i, 1],s=8,lw=1,c= colour)
         

    #Plot means if they are given
    if len(medoidInd) > 0:
        ax.scatter(dataset[medoidInd, 0], dataset[medoidInd, 1],s=8,lw=6, c=colours[:len(medoidInd)])
    fig.tight_layout()
    plt.show()
    
cluster_plots(data, colours =clusters, title='Unclustered data')
    
#distance calculator
def distance(X,Y):
    #Return the Euclidean distance between X and Y
    return np.linalg.norm(X-Y)

#Given dataset and indices of means, the function updates the clusters of the objects in the dataset
def assign(meansInd, dataset, clusters):
    numOfObjects = len(dataset)
    k = len(meansInd)
    
    #for every object in the dataset
    for i in range(numOfObjects):
        X = dataset[i]
        #find the closest medoid
        medoidIndOfX = -1;
        distanceToClosestMedoid = np.Inf;
        for index in meansInd:
            currentMedoid = dataset[index]
            dist = distance(X, currentMedoid)
            if dist < distanceToClosestMedoid:
                #Found closer medoid. Store information about it
                distanceToClosestMedoid = dist
                medoidIndOfX = index
        #assign to X its closest medoid
        clusters[i,0] = int(medoidIndOfX)
        
#Compute the objective fuction for the given dataset and the set of means
def objectiveFunc(meansInd, dataset):
    numOfObjects = len(dataset)
    
    clusters = np.zeros((len(dataset),2))
    
    #assign objects to closest means
    assign(meansInd, dataset, clusters)
    
    #compute the objective function: the total sum of distances from objects to their means
    obj = 0
    for i in range(numOfObjects):
        obj = obj + distance(dataset[i,:-1], dataset[int(clusters[i,0]),:-1])
    
    return obj

#change this to change number of clusters
Maxk = 9

BCubed = np.zeros(Maxk)    
Recall = np.zeros(Maxk) 
FScore = np.zeros(Maxk) 

def kMeans(k, dataset, clusters, maxIter=10):
    numOfObjects = len(dataset)
    
    ###Initialisation phase
    #Generate indices for initial means
    np.random.seed(45)
    meansInd = np.random.choice(numOfObjects, k, replace=False)
        
    #Make initial assignment of objects to the clusters
    assign(meansInd, dataset,clusters)
    
    #plot initial assignment
    cluster_plots(dataset, meansInd, clusters, title='Initial cluster assignment')
    
    ###Optimisation+Initialisation phases
    #Create temporary arrays with cluster and medoid indices for optimisation purposes
    tempClusters = np.copy(clusters)
    tempmeansInd = np.copy(meansInd)
    bestObjective = objectiveFunc(tempmeansInd, dataset)
    print('Initial objective function: %.2f' % bestObjective, ';    Initial Mean indices: ', meansInd)
    isObjectiveImproved = False

    
    for i in range(maxIter):
        isObjectiveImproved = False
        bestPair = np.zeros(2)
        #for each pair of (x,Y) where Y is a medoid and X is a non-medoid object from the dataset
        for y in range(k):
            for x in range(numOfObjects):
                if not x in tempmeansInd:
                    #put X into the set of means and instead of Y
                    tempmeansInd[y] = x
                    #and compute the value of the objective function for the new set of means
                    tempObjective = objectiveFunc(tempmeansInd, dataset)
                    #if the objective improved
                    if tempObjective < bestObjective:
                        #then update current best pair (X,Y) and current best objective
                        isObjectiveImproved = True
                        bestPair[0] = int(x);
                        bestPair[1] = int(y);
                        bestObjective = tempObjective
                    tempmeansInd[y] = meansInd[y]
                                            
        #If the objectvie function has improved in the current iteration, then
        if isObjectiveImproved:
            #update the set of means according to the best pair (X,Y)
            meansInd[int(bestPair[1])] = bestPair[0];
            #reassign objects to new means
            assign(meansInd, dataset, clusters)
            bestObjective = objectiveFunc(meansInd, dataset)
            #and plot the current clustering           
            cluster_plots(dataset, meansInd, clusters, title='Clustering improvement')
            print('Objective function: %.2f' % bestObjective, ';    Mean indices: ', meansInd)
        else:
            #otherwise stop clustering (we reached a local optimum)
            break
        
    #Add code to compute bcubed, precison, recall here            
    BCubedArray = np.zeros(len(dataset))
    RecallArray = np.zeros(len(dataset))
    FscoreArray = np.zeros(len(dataset))
    
    #For each item in the cluster
    for i in range(len(dataset)):
        NumberItemsWithMyLabelInCluster = 0
        NumberItemsInMyCluster = 0
        NumberItemsWithMyLabel = 0
        for j in range(len(dataset)):
            if(clusters[j,0] == clusters[i,0]):
                NumberItemsInMyCluster += 1
            if(clusters[j,1] == clusters[i,1]):
                NumberItemsWithMyLabel += 1
            if(clusters[j,1] == clusters[i,1] and clusters[j,0] == clusters[i,0]):
                NumberItemsWithMyLabelInCluster += 1
        
        BCubedArray[i] = NumberItemsWithMyLabelInCluster/NumberItemsInMyCluster
        RecallArray[i] = NumberItemsWithMyLabelInCluster/NumberItemsWithMyLabel
        FscoreArray = (2 * BCubedArray[i] * RecallArray[i]) / (BCubedArray[i] + RecallArray[i])
    
    BCubed[k-1] = np.average(BCubedArray)
    Recall[k-1] = np.average(RecallArray)
    FScore[k-1] = np.average(FscoreArray)
        
        

def Accuracy (k, BCubed, Recall, title):
    fig,ax = plt.subplots()
    fig.set_size_inches(5,5)
    ax.set_title(title,fontsize=10)
    ax.set_xlabel("Number of Clusters",fontsize=14)
    ax.set_ylabel("Value of Accuracy Measure",fontsize=14)

    ax.set_xlim(0.9, k+0.1)
    ax.set_ylim(0,1.1)
    ax.plot(range(1, k+1), BCubed,lw=1,c= 'red', marker='o')
    ax.plot(range(1, k+1), Recall,lw=1,c= 'blue', marker='o')
    ax.plot(range(1, k+1), FScore,lw=1,c= 'green', marker='o')
    ax.legend( ['BCubed', 'Recall', 'FScore'])
    
    #final plot
    fig.tight_layout()
    plt.show()


#main

for i in range(Maxk):
    kMeans(i+1,data,clusters)

Accuracy(Maxk, BCubed, Recall, "BCubed, Recall and F-Score over k clusters using K-Means Clustering")
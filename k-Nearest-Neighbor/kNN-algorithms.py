
#!/usr/bin/env python
#Handling Data
import csv
import random
"""
with open('iris.data') as csv_file:
    csv_reader=csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        print( ','.join(row))
"""
#split dataset
def loadDataset(filename,split,trainingSet=[],testingSet=[]):
    with open(filename) as csv_file:
        lines=csv.reader(csv_file)
        dataset=list(lines)
    for x in range(len(dataset)-1):
        for y in range(4):
            dataset[x][y]=float(dataset[x][y])
            if random.random()< split:
                trainingSet.append(dataset[x])
            else:
                testingSet.append(dataset[x])
#Downloading to local directory
trainingSet=[]
testingSet=[]
loadDataset('iris.data',0.66,trainingSet,testingSet)
print('Train:'+repr(len(trainingSet)))
print('Test:' + repr(len(testingSet)))

#similarity
import math
def euclideanDistance(instance1,instance2, length):
    distance=0
    for x in range(length):
        distance +=pow((instance1[x]-instance2[x]),2)
        return math.pow(distance,0.5)
#Neighbors
import operator
def getNeighbors(trainingSet, testInstance, k):
    distances=[]
    length= len(testInstance)-1
    for x in range(len(trainingSet)):
        dist= euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
        distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
#Response
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response]= 1
    sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]
#Accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
#main
KK=[3,5,10]
def main():

    # prepare data
    trainingSet=[]
    testingSet=[]
    split = 0.67
    loadDataset('iris.data', split, trainingSet, testingSet)
    print ('Train set: ' + str(len(trainingSet)))
    print ('Test set: ' + str(len(testingSet)))
    for k in KK:
        predictions=[]
        for x in range(len(testingSet)):
            neighbors=getNeighbors(trainingSet, testingSet[x], k)
            result=getResponse(neighbors)
            predictions.append(result)
            #print('> predicted=' + repr(result) + ', actual=' + repr(testingSet[x][-1]))
        accuracy=getAccuracy(testingSet,predictions)
        print('kNN=',k,'  ', 'Accuracy: ' + repr(accuracy) + '%')

main()
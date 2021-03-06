import numpy as np
import scipy as sci
import scipy.misc as scimisc
from scipy.special import gammaln

import csv
import operator
import sklearn.metrics

total_users = []
total_artists = []
total_freqs = []
f=open('./lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv')
for i in range(500000):
	line=f.next().split('\t')
	total_users.append(line[0])
	total_artists.append(line[2])
	total_freqs.append(line[3])

print "raw data loaded"

users_1k = []
artists_1k = []
freqs_1k = []
count = 0
for i in range(len(total_users)):
	users_1k.append(total_users[i])
	artists_1k.append(total_artists[i])
	freqs_1k.append(total_freqs[i])
	if total_users[i] != total_users[i-1]:
		count += 1
	if count == 10000:
		break

print "created 1k data"

# put users in dict: key=user, val=index

users = {}
user_count = 0

for i in range(len(users_1k)):
	if users_1k[i] not in users:
		users[users_1k[i]] = user_count
		user_count += 1

print "user_count:", user_count

# get top 1000 artists by number of users

NUM_ARTISTS = 1000

artists = {}

for i in range(len(users_1k)):
 	if artists_1k[i] not in artists:
 		artists[artists_1k[i]] = 1
 	else:
 		artists[artists_1k[i]] += 1


sorted_artists = sorted(artists.items(), key=operator.itemgetter(1), reverse=True)

new_artists = []

for i in range(NUM_ARTISTS):
    new_artists.append(sorted_artists[i][0])

artists = {}
artist_count = 0

for i in range(len(new_artists)):
    if new_artists[i] not in artists:
        artists[new_artists[i]] = artist_count
        artist_count += 1

data = np.genfromtxt('/Users/nabeelsarwar/Documents/PrincetonJuniorSpring/COS424/HW/FinalProjectGit2/cos424_final/data.txt')

NUM_VARIABLES = data.shape[1]

#K clusters
K = 10

clusterMean = []

multinomial = np.zeros(K)

for i in range(K):
    multinomial[i] = 1.0/K

print multinomial
#initialize clusters randomly
for i in range(K):
    mean = np.zeros(NUM_VARIABLES)
    for j in range(NUM_VARIABLES):
        mean[j] = np.random.randint(1, high=10)
    
    clusterMean.append(mean)

print 'Created Means'

#this is a matrix of the condition probabilities
#number of samples by number of clusters
conditionals = np.empty((data.shape[0], K))

print 'Created Conditionals'

def PoissonProbability(value, mean):
    if value==0:
        return np.exp(-mean)
    if mean==0:
        return -gammaln(value)
    return value * np.log(mean) - mean - gammaln(value)

def PoissonProbabilityVector(vector, mean):
    values = np.zeros(vector.shape[0])
    for i in range(vector.shape[0]):
        values[i] = PoissonProbability(vector[i], mean[i])
    return values
    
def updateConditions():
    for i in range(conditionals.shape[0]):
        rowsumForBayes = np.zeros(K)
        for j in range(conditionals.shape[1]):
            if (data[i, :].shape[0] != clusterMean[j].shape[0]):
                print 'size mismatch 1'
            rowsumForBayes[j] = np.log(multinomial[j]) + np.sum(PoissonProbabilityVector(data[i, :], clusterMean[j]))
        rowsumForBayes = scimisc.logsumexp(rowsumForBayes)  
        for j in range(conditionals.shape[1]):
            conditionals[i, j] = np.exp(np.log(multinomial[j]) + np.sum(PoissonProbabilityVector(data[i,:], clusterMean[j])) - rowsumForBayes)
        if not np.isclose(np.sum(conditionals[i, :]), 1):
            return 'Probabilities not one'
            



def updateMultiNomial():
    for i in range(multinomial.shape[0]):
        multinomial[i] = np.sum(conditionals[:, i])/data.shape[0]

def updateMeans():
    for i in range(len(clusterMean)):
        numerator = 0 
        for j in range(conditionals.shape[0]):
            #you want j, i for real in here as in the notation of the book
            numerator = numerator + conditionals[j, i] * data[j, :]
        if (numerator.shape[0] != clusterMean[i].shape[0]):
            print 'Size mismatch 2 '
        clusterMean[i] = numerator/(multinomial[i] * data.shape[0])

counter = 0
multinomialDifference = 1

while counter < 10:
    oldMulti = multinomial
    updateConditions()
    updateMultiNomial()
    updateMeans()
    multinomialDifference = np.sum(np.abs(multinomial - oldMulti))
    counter = counter + 1
    print 'Loop {0}'.format(counter)
    
clusterMean = np.array(clusterMean)
print clusterMean.shape[0]
np.savetxt('clusterMean.txt', clusterMean, fmt='%s')

#now we need to output the max conditional for each row
whichGroupMax = np.zeros(data.shape[0])

#cluster for eachad ta
whichGroupMax = np.argmax(conditionals, axis = 1)
print 'Which groups'
print whichGroupMax

np.savetxt('clustersOfTrainData.txt', whichGroupMax, fmt='%s')

print 'multinomial'
print multinomial
np.savetxt('PoissonMixtureClassProbabilities.txt', multinomial, fmt='%s')

#what does each cluster have
clusterIndices = []
for i in range(len(clusterMean)):
    clusterIndices.append(np.argsort(clusterMean[i, : ])[-5:])


def getArtist(index):
    for key in artists:
        if (artists.get(key) == index):
            return key


clusterIndices = np.array(clusterIndices)

artistsForEachCluster = []
for i in range(clusterIndices.shape[0]):
    cluster = []
    for j in range(clusterIndices.shape[1]):
        cluster.append(str(getArtist(clusterIndices[i, j])) + ',')
    cluster = np.array(cluster)
    artistsForEachCluster.append(cluster)

artistsForEachCluster = np.array(artistsForEachCluster)

np.savetxt('artistsFromPoissonMixtureModel.txt', artistsForEachCluster, fmt='%s')

shadowscore = sklearn.metrics.silhouette_score(data, whichGroupMax)

print 'Silhouette score: {0}'.format(shadowscore)

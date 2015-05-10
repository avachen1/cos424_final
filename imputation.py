import numpy as np
import csv
import operator
from random import randint

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

print "artist_count:", artist_count


# fill in matrix (user_count x artist_count) with frequencies

matrix = np.zeros([user_count, artist_count],dtype=np.int)
matrix_heldout = np.zeros([user_count, artist_count])
for i in range(len(users_1k)):
	if (artists_1k[i] in artists):
		x = randint(0,9)
		if (x == 0):
			matrix_heldout[users[users_1k[i]]][artists[artists_1k[i]]] = freqs_1k[i]		
		else:
			matrix[users[users_1k[i]]][artists[artists_1k[i]]] = int(freqs_1k[i])

print "matrix created"

counts = []
for i in range(0,100):
	counts.append(0)

delete = []
for i in range(user_count):
	count = 0
	for j in range(artist_count):
		if (matrix[i,j] != 0):
			count += 1
	if (count < 2):
		delete.append(i)

new_matrix = np.delete(matrix,delete,0)
new_matrix_heldout = np.delete(matrix_heldout,delete,0)

heldout_users = []
heldout_artists = []
heldout_freqs = []

count = 0

for i in range(new_matrix_heldout.shape[0]):
	for j in range(new_matrix_heldout.shape[1]):
		if (new_matrix_heldout[i,j] != 0):
			count += 1
			heldout_users.append(i)
			heldout_artists.append(j)
			heldout_freqs.append(new_matrix_heldout[i,j])

print count
print len(heldout_users)

heldout_triples = np.zeros([count,3],dtype=np.int)

for i in range(count):
	heldout_triples[i,0] = int(heldout_users[i])
	heldout_triples[i,1] = int(heldout_artists[i])
	heldout_triples[i,2] = int(heldout_freqs[i])

print heldout_triples[5]

print heldout_triples.shape

print new_matrix.shape

zero_users = []
zero_artists = []
count = 0
for i in range(new_matrix.shape[0]):
	for j in range(new_matrix.shape[1]):
		x = randint(0,300)
		if (x == 0):
			if ((new_matrix[i,j] == 0) and (new_matrix_heldout[i,j] == 0)):
				count += 1
				zero_users.append(i)
				zero_artists.append(j)

print count

zero_triples = np.zeros([count,3],dtype=np.int)

print zero_triples.shape
for i in range(count):
	zero_triples[i,0] = int(zero_users[i])
	zero_triples[i,1] = int(zero_artists[i])
	zero_triples[i,2] = int(0)

im_test = np.concatenate((heldout_triples, zero_triples))

print im_test.shape

np.savetxt('imputation_train.txt', new_matrix, fmt='%s')
np.savetxt('imputation_test.txt', im_test, fmt='%s')







import numpy as np
import csv
import operator

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




# put artists in dict: key=artist, val=index

# artists = {}
# artist_count = 0

# for i in range(len(artists_1k)):
#	if artists_1k[i] not in artists:
#		artists[artists_1k[i]] = artist_count
#		artist_count += 1

# print "artist_count:", artist_count




# fill in matrix (user_count x artist_count) with frequencies

matrix = np.zeros([user_count, artist_count])
for i in range(len(users_1k)):
	if (artists_1k[i] in artists):
		matrix[users[users_1k[i]]][artists[artists_1k[i]]] = freqs_1k[i]

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

print new_matrix.shape

binary_matrix = np.zeros([len(new_matrix), artist_count])

for i in range(len(new_matrix)):
	for j in range(artist_count):
		if new_matrix[i,j] > 0:
			binary_matrix[i,j] = 1


# don't need to remake data over and over again
np.savetxt('data.txt', new_matrix, fmt='%s')
np.savetxt('binary.txt', binary_matrix, fmt='%s')


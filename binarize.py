import numpy as np
import sys

if sys.argv[2] == "matrix":
	train = np.genfromtxt(sys.argv[1])
	binary_train = np.zeros([len(train), len(train[0])], dtype=np.int)
	for i in range(len(train)):
		for j in range(len(train[0])):
			if int(train[i,j]) > 0:
				binary_train[i,j] = 1
			elif int(train[i,j]) == 0:
				binary_train[i,j] = 0
elif sys.argv[2] == "tuples":
	f = open(sys.argv[1], 'r')
	binary_train = np.array([l.split() for l in f.readlines()])
	for i in range(len(binary_train)):
		if int(binary_train[i,2]) > 0:
			binary_train[i,2] = 1
		elif int(binary_train[i,2]) == 0:
			binary_train[i,2] = 0

if sys.argv[4] == "matrix":
	test = np.genfromtxt(sys.argv[3])
	binary_test = np.zeros([len(test), len(test[0])], dtype=np.int)
	for i in range(len(test)):
		for j in range(len(test[0])):
			if int(test[i,j]) > 0:
				binary_test[i,j] = 1
			elif int(test[i,j]) == 0:
				binary_test[i,j] = 0
elif sys.argv[4] == "tuples":
	f = open(sys.argv[3], 'r')
	binary_test = np.array([l.split() for l in f.readlines()])
	for i in range(len(binary_test)):
		if int(binary_test[i,2]) > 0:
			binary_test[i,2] = 1
		elif int(binary_test[i,2]) == 0:
			binary_test[i,2] = 0

np.savetxt(sys.argv[1][:-4]+'_binary.txt', binary_train, fmt='%s')
np.savetxt(sys.argv[3][:-4]+'_binary.txt', binary_test, fmt='%s')
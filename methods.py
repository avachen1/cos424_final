from scipy import sparse
import numpy as np
from sklearn.decomposition import TruncatedSVD, RandomizedPCA, NMF
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from functools import partial
import math
from sklearn.mixture import GMM

def nmf(X, n, binary, d):
	name = "nmf" + str(n)
	if binary:
		name = name + "_binary_" + d
	print name
	model = NMF(n_components = n)
	model.fit(X)
	A = model.components_
	A_T = A.transpose()
	Z = model.fit_transform(X)
	def get_prob(i, j):
		return np.sum((Z[i,:]*A[:,j]))
	display_results(name, get_prob, binary, d)

def svd(X, n, binary, d):
	name = "svd" + str(n)
	if binary:
		name = name + "_binary_" + d
	print name
	model = TruncatedSVD(n_components = n)
	model.fit(X)
	print(model.explained_variance_ratio_)
	print "total explained variance", model.explained_variance_ratio_.sum()
	A = model.components_
	A_T = A.transpose()
	Z = model.fit_transform(X)
	def get_prob(i, j):
		return np.sum((Z[i,:]*A[:,j]))
	display_results(name, get_prob, binary, d)

def kmeans(X, n, binary, d):
	name = "kmeans" + str(n)
	if binary:
		name = name + "_binary_" + d
	print name
	model = KMeans(n_clusters = n)
	labels = model.fit_predict(X)
	def get_prob(i,j):
		return model.cluster_centers_[labels[i]][j]
	display_results(name, get_prob, binary, d)

def gmm(X, n, binary, d):
	name = "gmm" + str(n)
	if binary:
		name = name + "_binary_" + d
	print name
	model = GMM(n_components = n)
	model.fit(X)
	labels = model.predict(X)
	def get_prob(i,j):
		return model.means_[labels[i]][j]
	display_results(name, get_prob, binary, d)

def display_results(name, get_prob, binary, d):
	if d == "split":
		if binary:
			test = np.genfromtxt("./test_binary.txt")
		else:
			test = np.genfromtxt("./test.txt")
		out = open(name+"_probs.txt", "w")
		probs = []
		trues = []
		preds = []
		for i in range(len(test)):
			for j in range(len(test[0])):
				# user = i
				# artist = j
				value = int(test[i,j])
				prob = get_prob(i,j)
				# print prob
				if binary:
					if prob > 1:
						prob = 1
					elif prob < 0:
						prob = 0
				if prob < 0.5:
					pred = 0
				else: pred = 1
				out.write(str(prob)+"\n")
				preds.append(pred)
				trues.append(value)
				probs.append(prob)
		out.close()

		precision, recall, thresholds = precision_recall_curve(trues, probs, pos_label =1)
		pr_auc = auc(recall, precision)
		print "precision recall AUC", pr_auc 
		plt.plot(recall, precision, label=name)
		rand = metrics.adjusted_rand_score(trues, preds)
		print "adjusted rand score", rand
		mi = metrics.adjusted_mutual_info_score(trues, preds)
		print "adjusted mutual info score", mi
		homogeneity = metrics.homogeneity_score(trues, preds)  
		print "homogeneity score", homogeneity
		completeness = metrics.completeness_score(trues, preds)  
		print "completeness score", completeness
		v_measure = metrics.v_measure_score(trues, preds)  
		print "v-measure score", v_measure

	elif d == "imputation":
		if binary:
			f_test = open('./imputation_test_binary.txt', 'r')
		else:
			f_test = open('./imputation_test.txt', 'r')
		f_out = open(name+"_probs.txt", "w")
		test = np.array([l.split() for l in f_test.readlines()])
		probs = []
		trues = [] 
		preds = []
		for l in test:
			i = int(l[0])
			j = int(l[1])
			value = int(l[2])
			prob = get_prob(i,j)
			if binary:
				if prob > 1:
					prob = 1
				elif prob < 0:
					prob = 0
			if prob < 0.5:
				pred = 0
			else: pred = 1
			f_out.write(str(prob)+"\n")
			preds.append(pred)
			trues.append(value)
			probs.append(prob)
		f_test.close()
		f_out.close()

		precision, recall, thresholds = precision_recall_curve(trues, probs, pos_label =1)
		pr_auc = auc(recall, precision)
		print "precision recall AUC", pr_auc 
		plt.plot(recall, precision, label=name)
		rand = metrics.adjusted_rand_score(trues, preds)
		print "adjusted rand score", rand
		mi = metrics.adjusted_mutual_info_score(trues, preds)
		print "adjusted mutual info score", mi
		homogeneity = metrics.homogeneity_score(trues, preds)  
		print "homogeneity score", homogeneity
		completeness = metrics.completeness_score(trues, preds)  
		print "completeness score", completeness
		v_measure = metrics.v_measure_score(trues, preds)  
		print "v-measure score", v_measure

binaries = [True]
# binaries = [True, False]
data = ["imputation"]
# data = ["split", "imputation"]
for binary in binaries:
	for d in data:
		if binary:
			# data = np.genfromtext("./data.txt")
			if d == "split":
				train = np.genfromtxt("./train_binary.txt")
			elif d == "imputation":
				train = np.genfromtxt("./imputation_train_binary.txt")
		else:
			# data = np.genfromtext("./binary.txt")
			if d == "split":
				train = np.genfromtxt("./train.txt")
			elif d == "imputation":
				train = np.genfromtxt("./imputation_train.txt")
		nmf(train, 20, binary, d)
		svd(train, 20, binary, d)
		kmeans(train, 20, binary, d)
		gmm(train, 20, binary, d)


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
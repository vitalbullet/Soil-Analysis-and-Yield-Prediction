from sys import stdin,stdout

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

trainingData = pd.read_csv('DatasetWithCategory.csv')
testingData = pd.read_csv('TestingDataSet.csv')

#print(trainingData)
trainfeature = trainingData.values[:,1:-1]
testfeature = testingData.values[:,1:]
#print(trainfeature)
#print(testfeature)
#print(np.concatenate((trainfeature,testfeature),axis=0))
traincat = trainingData.Category
#print(traincat)


def knn(n=3):
	global trainfeature,testfeature,traincat
	classifier = KNeighborsClassifier(n_neighbors=n)
	classifier.fit(trainfeature,traincat)
	prediction = classifier.predict(testfeature)
	print('*'*204)
	print('{:^204s}'.format('K-Nearest Neighbor Algorithm'))
	print('\n\n')
	print('Predictions:\n\n')
	for i in range(1,len(prediction)):
		print(str(i)+'.',prediction[i-1])
	print('\n\n')
	

def naiveBayes():
	global trainfeature,testfeature,traincat
	classifier = GaussianNB()
	classifier.fit(trainfeature,traincat)
	prediction = classifier.predict(testfeature)
	print('*'*204)
	print('{:^204s}'.format('Naive Bayes Algorithm'))
	print('\n\n')
	print('Predictions:\n\n')
	for i in range(1,len(prediction)):
		print(str(i)+'.',prediction[i-1])
	print('\n\n')
	
def supportVectorMachine():
	global trainfeature,testfeature,traincat
	classifier = svm.SVC()
	classifier.fit(trainfeature,traincat)
	prediction = classifier.predict(testfeature)
	print('*'*204)
	print('{:^204s}'.format('Support Vector Machine Classifier'))
	print('\n\n')
	print('Predictions:\n\n')
	for i in range(1,len(prediction)):
		print(str(i)+'.',prediction[i-1])
	print('\n\n')
	
def kmeans(n=3):
	global trainfeature,testfeature	
	estimator = KMeans(n)
	estimator.fit(np.concatenate((trainfeature,testfeature),axis=0))
	prediction = estimator.labels_
	print('*'*204)
	print('{:^204s}'.format('K-Means Clustering Algorithm'))
	print('\n\n')
	print('Result after Clustering:\n\n')
	for i in range(1,len(prediction)):
		print(str(i)+'.','Cluster #:\t\t',prediction[i-1]+1)
	print('\n\n')
	
def hcluster():
	global trainfeature,testfeature
	linkage_matrix = linkage(np.concatenate((trainfeature,testfeature),axis=0),'single')
	dendogram = dendrogram(linkage_matrix,truncate_mode='none')
	plt.title('Hierarchical Clustering')
	plt.show()	

def Sup():
	while True:
		ch = menu(['Supervised Learning','K-Nearest Neighbor Algorithm','Naive Bayes Algorithm','Support Vector Machine Classifier','Back'])
		print('\n\n')		
		if ch==4:
			return
		elif ch==1:
			
			knn(3)
		elif ch==2:
			naiveBayes()
		elif ch==3:
			supportVectorMachine()	
		else:
			stdout.write('\nINVALID RESPONSE, TRY AGAIN .........\n\n')

def Usup():
	while True:
		ch = menu(['Unsupervised Learning','K Means Clustering Algorithm','Hierarchical Clustering Algorithm','Back'])
		print('\n\n')		
		if ch==3:
			return
		elif ch==1:
			kmeans(3)
		elif ch==2:
			hcluster()
		else:
			stdout.write('\nINVALID RESPONSE, TRY AGAIN .........\n\n')

def menu(x):
	print('*'*204)
	print('{:^204s}'.format(x[0]))
	print('\n\n')
	for i in range(1,len(x)):
		print(str(i)+'.',x[i])
	stdout.write('\n\nEnter your Choice:\t')
	return int(stdin.readline())

def main():
	while True:
		ch = menu(['Soil Analysis and Yield Prediction','Supervised Learning','Unsupervised Learning','Exit'])
		print('\n\n')		
		if ch==3:
			break
		elif ch==1:
			Sup()
		elif ch==2:
			Usup()
		else:
			stdout.write('\nINVALID RESPONSE, TRY AGAIN .........\n\n')
		#print('{:^204s}'.format('*'*204))
	print('\n\n')
	print('{:^204s}'.format('Authors:\tKshitij Jaiswal, Vibhav , Gaurav Khattar\n'))
	print('{:^204s}'.format('THANKS YOU FOR USING OUR SOFTWARE'))

if __name__=='__main__':
	main()

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sys
import os
import time
sys.path.append('../../pyTsetlinMachineParallel/')
from tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
import re
import string
import pickle
import csv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from time import time


oplabels=['0','1']
qr=['query_word']
gram_base=["pos_", "tag_", "ent_type_", "is_alpha", "is_stop", "is_digit", "is_lower", "is_upper","is_punct", "is_left_punct", "is_right_punct", "is_bracket", "is_quote", "dep_", "head.pos_", "head.head.pos_"]
gram_base+=qr
exf=['text','word_idx','label']

CLAUSES=1000
T=1500
s=27.5
weighting = True
motif_length=7
training_epoch=100
RUNS=20


def find_uniques_length(df):
	uniques=[]
	num_columns=df.shape[2]
	#df[:,:,0]
	for col in range(len(gram_base)):
		uset=set(df[:,:,col].flatten())
		if uniques==[]:
			uniques=[uset]
		else:
			uniques.append(uset)
	return uniques

def binarize(df, list_uniques, list_columns):
	temp_cols=[]
	sum_size=np.sum([len(s) for s in list_uniques])
	newX=np.zeros((df.shape[0], df.shape[1], sum_size), dtype=np.int32)
	startind=0
	for col in range(len(gram_base)):
		ul=list(list_uniques[col])
		arr=df[:,:,col].flatten()
		tempx=[[1]*(ul.index(arr[pos])+1)+[0]*(len(ul)-(ul.index(arr[pos])+1)) for pos in range(len(arr))]
		tempx=np.reshape(tempx,(df.shape[0],df.shape[1],len(ul)))
		endind=startind+len(ul)
		newX[:,:,startind:endind]=tempx
		startind=endind
	return newX	

with open('../../pickles/spacy/nonbinarized_features_sentence_train_glove.pkl','rb') as f:
	glove_features_train=pickle.load(f)
with open('../../pickles/spacy/nonbinarized_features_sentence_train_gram.pkl','rb') as f:
	grammar_features_train=pickle.load(f)
	
with open('../../pickles/spacy/nonbinarized_features_sentence_test_glove.pkl','rb') as f:
	glove_features_test=pickle.load(f)
with open('../../pickles/spacy/nonbinarized_features_sentence_test_gram.pkl','rb') as f:
	grammar_features_test=pickle.load(f)

with open('../../pickles/spacy/nonbinarized_features_sentence_train_labels.pkl','rb') as f:
	labels_train=pickle.load(f)
with open('../../pickles/spacy/nonbinarized_features_sentence_test_labels.pkl','rb') as f:
	labels_test=pickle.load(f)
	
label_set=list(set(labels_train+labels_test))
labels_train=[label_set.index(ls) for ls in labels_train]
labels_test=[label_set.index(ls) for ls in labels_test]


grammar_features_train=np.asarray(grammar_features_train)
grammar_features_train=grammar_features_train[:,:,2:]

grammar_features_test=np.asarray(grammar_features_test)
grammar_features_test=grammar_features_test[:,:,2:]

glove_features_train=np.asarray(glove_features_train)
glove_features_train=glove_features_train[:,:,2:]

glove_features_test=np.asarray(glove_features_test)
glove_features_test=glove_features_test[:,:,2:]

print('glove train', glove_features_train.shape)
print('grammar train', grammar_features_train.shape)

#########Grammar##########
print("Grammar")

combo_train=grammar_features_train
combo_test=grammar_features_test
				 
list_of_uniques=find_uniques_length(combo_train)
print(list_of_uniques)
#print('len col',len(baselist_gram))

usum=np.sum([len(s) for s in list_of_uniques])
print('sum', usum)	

Xtrain=binarize(combo_train, list_of_uniques, gram_base)
Xtest=binarize(combo_test, list_of_uniques, gram_base)

print('binarized train',Xtrain.shape)
print('binarized test',Xtest.shape)

X_train = Xtrain.reshape((Xtrain.shape[0],Xtrain.shape[1],1,usum))
X_test = Xtest.reshape((Xtest.shape[0],Xtest.shape[1],1,usum))

print('reshaped train',X_train.shape)
print('reshaped test',X_test.shape)

#np.save('x_train_conv', Xtrain)
#np.save('x_test_conv', Xtest)

'''# Setup
tm = MultiClassConvolutionalTsetlinMachine2D(CLAUSES, T, s, (motif_length, 1), weighted_clauses=weighting)
#labels_test_indx=np.where(labels_test==1)
#labels_train_indx=np.where(labels_train==1)

acc=[]

# Training
for i in range(RUNS):
	print(i)
	start_training = time()
	tm.fit(X_train, labels_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	res_test=tm.predict(X_test)
	res_train=tm.predict(X_train) 
		
	result_test = 100*(res_test == labels_test).mean()
	result_train = 100*(res_train == labels_train).mean()
	prf_test=precision_recall_fscore_support(res_test, labels_test, average='macro')
	prf_test=[str(p) for p in prf_test]
	prf_test=' '.join(prf_test)
	prf_train=precision_recall_fscore_support(res_train, labels_train, average='macro')
	prf_train=[str(p) for p in prf_train]
	prf_train=' '.join(prf_train)

	stop_testing = time()

	print("\n\n#%d Testing Accuracy: %.2f%% Training Accuracy: %.2f%% Training Time: %.2fs Testing Time: %.2fs" % (i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
	print("\n#Testing PRF: %s%%\nTraining PRF: %s%%" % (prf_test, prf_train))
'''
#########Glove##########
print("\n\nGlove")

combo_train=glove_features_train
combo_test=glove_features_test

print('combo train',combo_train.shape)
print('combo test',combo_test.shape)

X_train2 = combo_train
X_test2 = combo_test
				 
# Setup
tm = MultiClassConvolutionalTsetlinMachine2D(CLAUSES, T, s, (motif_length, 1), weighted_clauses=weighting)
labels_test_indx=np.where(labels_test==1)
labels_train_indx=np.where(labels_train==1)

acc=[]

# Training
for i in range(RUNS):
	print(i)
	start_training = time()
	tm.fit(X_train2, labels_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	res_test=tm.predict(X_test2)
	res_train=tm.predict(X_train2) 
		
	result_test = 100*(res_test == labels_test).mean()
	result_train = 100*(res_train == labels_train).mean()
	prf_test=precision_recall_fscore_support(res_test, labels_test, average='macro')
	prf_test=[str(p) for p in prf_test]
	prf_test=' '.join(prf_test)
	prf_train=precision_recall_fscore_support(res_train, labels_train, average='macro')
	prf_train=[str(p) for p in prf_train]
	prf_train=' '.join(prf_train)

	stop_testing = time()

	print("\n\n#%d Testing Accuracy: %.2f%% Training Accuracy: %.2f%% Training Time: %.2fs Testing Time: %.2fs" % (i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
	print("\n#Testing PRF: %s%%\nTraining PRF: %s%%" % (prf_test, prf_train))
	#print("\nActual Testing Accuracy: %.2f%% Training Accuracy: %.2f%%" % (result_test2, result_train2))
	acc.append(result_test2)
vjhv	
#########Combo##########
print("\n\nCombo")

X_train3 = np.concatenate((X_train,X_train2), axis=3)
X_test3 = np.concatenate((X_test,X_test2), axis=3)

print('reshaped train',X_train3.shape)
print('reshaped test',X_test3.shape)
				 
# Setup
tm = MultiClassConvolutionalTsetlinMachine2D(CLAUSES, T, s, (motif_length, 1), weighted_clauses=weighting)
labels_test_indx=np.where(labels_test==1)
labels_train_indx=np.where(labels_train==1)

acc=[]

# Training
for run in range(RUNS):
	print(run)
	start_training = time()
	tm.fit(X_train3, labels_train, epochs=training_epoch, incremental=True)
	stop_training = time()

	start_testing = time()
	res_test=tm.predict(X_test3)
	res_train=tm.predict(X_train3) 
	stop_testing = time()

	result_test = 100*(res_test == labels_test).mean()
	result_train = 100*(res_train == labels_train).mean()

	prf_test=precision_recall_fscore_support(res_test, labels_test, average='macro')
	prf_train=precision_recall_fscore_support(res_train, labels_train, average='macro')
	prf_detail_test=precision_recall_fscore_support(res_test, labels_test, average=None)
	prf_detail_train=precision_recall_fscore_support(res_train, labels_train, average=None)
	prf_test=[str(p) for p in prf_test]
	prf_test=' '.join(prf_test)
	prf_train=[str(p) for p in prf_train]
	prf_train=' '.join(prf_train)


	print("\n\n#%d Convolutional Testing Accuracy: %.2f%% Training Accuracy: %.2f%% Training Time: %.2fs Testing Time: %.2fs" % (run+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
	#print("\nActual Testing Accuracy: %.2f%% Training Accuracy: %.2f%%" % (result_test2, result_train2))
	print("#Testing PRF: %s%%\nTraining PRF: %s%%" % (prf_test, prf_train))
	print("#Classwise Testing  & Training PRFS:\n")
	'''for clidx in range(len(oplabels)):
		print(oplabels[clidx]+": "+str(prf_detail_test[0][clidx])+" ; "+str(prf_detail_test[1][clidx])+" ; "+str(prf_detail_test[2][clidx])+" ; "+str(prf_detail_test[3][clidx])+" || "+str(prf_detail_train[0][clidx])+" ; "+str(prf_detail_train[1][clidx])+" ; "+str(prf_detail_train[2][clidx])+" ; "+str(prf_detail_train[3][clidx])+'\n')
	'''
	#print("\nActual Testing Accuracy: %.2f%% Training Accuracy: %.2f%%" % (result_test2, result_train2))
	acc.append(result_test2)

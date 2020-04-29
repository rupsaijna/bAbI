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
context_length=4
qr=['query_word']
gram_base=["pos_", "tag_", "ent_type_", "is_alpha", "is_stop", "is_digit", "is_lower", "is_upper","is_punct", "is_left_punct", "is_right_punct", "is_bracket", "is_quote", "dep_", "head.pos_", "head.head.pos_"]
gram_base+=qr
addendum_context=['_wb'+str(l) for l in range(context_length,0,-1)]+['_wt']+['_wa'+str(l) for l in range(1,context_length+1)]
exf=['text','word_idx','label']

CLAUSES=160
T=90
s=2.7
weighting = True
motif_length=7
training_epoch=35
RUNS=20


def find_uniques_length(df):
	uniques=[]
	columns=[]
	
	dfcols=df.columns
	for col in gram_base:
		this_cols=[col+ad for ad in addendum_context]
		uset=set(df[this_cols].values.T.ravel())
		#columns+=[col]
		if uniques==[]:
			uniques=[uset]
		else:
			uniques.append(uset)
	return uniques

def binarize(df, list_uniques, list_columns):
	temp_cols=[]
	sum_size=np.sum([len(s) for s in list_uniques])
	newX=np.zeros((df.shape[0], sum_size*len(addendum_context)), dtype=np.int32)
	startind=0
	for contextid in addendum_context:
		for colname_base in gram_base:
			colname=colname_base+contextid
			ul=list(list_uniques[gram_base.index(colname_base)])
			tempx=np.zeros((df.shape[0], len(ul)), dtype=np.int32)
			arr=df[colname].tolist()
			tempx=[[1]*(ul.index(arr[pos])+1)+[0]*(len(ul)-(ul.index(arr[pos])+1)) for pos in range(len(arr))]
			tempx=np.reshape(tempx,(df.shape[0], len(ul)))
			endind=startind+len(ul)
			#print('name,s,e',colname,startind,endind)
			temp_cols.append(colname)
			newX[:,startind:endind]=tempx
			startind=endind
	'''temp_cols=np.array(temp_cols)
	print(temp_cols.shape)
	t=temp_cols.reshape(len(addendum_context),1,len(gram_base))
	print(t.shape)
	print(t)'''
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


print('gram',len(grammar_features_train), len(grammar_features_train[0]), len(grammar_features_train[0][0]), len(labels_train))
grammar_features_train=np.asarray(grammar_features_train)
grammar_features_train=grammar_features_train[:,:,2:]

grammar_features_test=np.asarray(grammar_features_test)
grammar_features_test=grammar_features_test[:,:,2:]

print('glove',len(glove_features_train), len(glove_features_train[0]), len(glove_features_train[0][0]), len(labels_train))
#glove_features_train=np.asarray(glove_features_train)

glove_features_train_new=np.array((len(glove_features_train), len(glove_features_train[0]), len(glove_features_train[0][0]-2)))
for s in range(len(glove_features_train)):
	print(s)
	for w in range(len(glove_features_train[s])):
		glove_features_train_new[s][w]=np.array(glove_features_train[s][w][2:])

#glove_features_train=np.asarray(glove_features_train)
print('glove', glove_features_train_new.shape)
print('glove', glove_features_train_new[0])

#print('glove',len(glove_features_train), len(glove_features_train[0]), len(glove_features_train[0][0]), len(labels_train))

asd
print('glove',len(glove_features_train), len(glove_features_train[0]), len(glove_features_train[0][0]), len(labels_train))

#########Grammar##########
print("Grammar")

combo_train=grammar_features_train
combo_test=grammar_features_test
				 
list_of_uniques=find_uniques_length(combo_train, remheaders)
#print(list_of_uniques)
#print('len col',len(baselist_gram))

usum=np.sum([len(s) for s in list_of_uniques])
#print('sum', usum, 'totsum', usum*5)	

Xtrain=binarize(combo_train, list_of_uniques, gram_base)
Xtest=binarize(combo_test, list_of_uniques, gram_base)

print('binarized train',Xtrain.shape)
print('binarized test',Xtest.shape)

X_train = Xtrain.reshape((Xtrain.shape[0],len(addendum_context),1,usum))
X_test = Xtest.reshape((Xtest.shape[0],len(addendum_context),1,usum))

print('reshaped train',X_train.shape)
print('reshaped test',X_test.shape)

#np.save('x_train_conv', Xtrain)
#np.save('x_test_conv', Xtest)
				 
# Setup
tm = MultiClassConvolutionalTsetlinMachine2D(CLAUSES, T, s, (motif_length, 1), weighted_clauses=weighting)
labels_test_indx=np.where(labels_test==1)
labels_train_indx=np.where(labels_train==1)

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
	
	res_test_indx=np.where(res_test==1)
	res_train_indx=np.where(res_train==1)	      
	
	result_test2=100*len(set(list(res_test_indx[0])).intersection(set(list(labels_test_indx[0]))))/len(list(labels_test_indx[0]))
	result_train2=100*len(set(list(res_train_indx[0])).intersection(set(list(labels_train_indx[0]))))/len(list(labels_train_indx[0]))
	
	result_test = 100*(res_test == labels_test).mean()
	result_train = 100*(res_train == labels_train).mean()
	prf_test=precision_recall_fscore_support(res_test, labels_test, average='macro')
	prf_test=[str(p) for p in prf_test]
	prf_test=' '.join(prf_test)
	prf_train=precision_recall_fscore_support(res_train, labels_train, average='macro')
	prf_train=[str(p) for p in prf_train]
	prf_train=' '.join(prf_train)

	stop_testing = time()

	'''print("\n\n#%d Testing Accuracy: %.2f%% Training Accuracy: %.2f%% Training Time: %.2fs Testing Time: %.2fs" % (i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
	print("\n#Testing PRF: %s%%\nTraining PRF: %s%%" % (prf_test, prf_train))'''
	print("\nActual Testing Accuracy: %.2f%% Training Accuracy: %.2f%%" % (result_test2, result_train2))
	acc.append(result_test2)
	
print('Max Acc:', max(acc))
print('Min Acc:', min(acc))
print('Avg Acc:', sum(acc)/len(acc))

#########Glove##########
print("\n\nGlove")

combo_train=glove_features_train
combo_test=glove_features_test

remheaders=['text','label', 'word_idx']

a=set(combo_train.columns.tolist())
b=set(combo_test.columns.tolist())
combo_headers=list(a.intersection(b))
combo_headers=[ch for ch in combo_headers if ch not in remheaders]

print('combo train',combo_train.shape)
print('combo test',combo_test.shape)

combo_train=combo_train[combo_headers].to_numpy()
combo_test=combo_test[combo_headers].to_numpy()

print('\ncombo train',combo_train.shape)
print('combo test',combo_test.shape)

X_train2 = combo_train.reshape((combo_train.shape[0],(context_length*2+1),1,int(combo_train.shape[1]/(context_length*2+1))))
X_test2 = combo_test.reshape((combo_test.shape[0],(context_length*2+1),1,int(combo_train.shape[1]/(context_length*2+1))))

print('reshaped train',X_train2.shape)
print('reshaped test',X_test2.shape)
				 
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
	
	res_test_indx=np.where(res_test==1)
	res_train_indx=np.where(res_train==1)	      
	
	result_test2=100*len(set(list(res_test_indx[0])).intersection(set(list(labels_test_indx[0]))))/len(list(labels_test_indx[0]))
	result_train2=100*len(set(list(res_train_indx[0])).intersection(set(list(labels_train_indx[0]))))/len(list(labels_train_indx[0]))
		
	result_test = 100*(res_test == labels_test).mean()
	result_train = 100*(res_train == labels_train).mean()
	prf_test=precision_recall_fscore_support(res_test, labels_test, average='macro')
	prf_test=[str(p) for p in prf_test]
	prf_test=' '.join(prf_test)
	prf_train=precision_recall_fscore_support(res_train, labels_train, average='macro')
	prf_train=[str(p) for p in prf_train]
	prf_train=' '.join(prf_train)

	stop_testing = time()

	'''print("\n\n#%d Testing Accuracy: %.2f%% Training Accuracy: %.2f%% Training Time: %.2fs Testing Time: %.2fs" % (i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
	print("\n#Testing PRF: %s%%\nTraining PRF: %s%%" % (prf_test, prf_train))'''
	print("\nActual Testing Accuracy: %.2f%% Training Accuracy: %.2f%%" % (result_test2, result_train2))
	acc.append(result_test2)
	
print('Max Acc:', max(acc))
print('Min Acc:', min(acc))
print('Avg Acc:', sum(acc)/len(acc))


	
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
	
	res_test_indx=np.where(res_test==1)
	res_train_indx=np.where(res_train==1)	      
	
	result_test2=100*len(set(list(res_test_indx[0])).intersection(set(list(labels_test_indx[0]))))/len(list(labels_test_indx[0]))
	result_train2=100*len(set(list(res_train_indx[0])).intersection(set(list(labels_train_indx[0]))))/len(list(labels_train_indx[0]))
	

	result_test = 100*(res_test == labels_test).mean()
	result_train = 100*(res_train == labels_train).mean()
	
	'''if(sum(res_train)>0):
		pz=list(zip(grammar_features_train['text'],grammar_features_train['label'],res_train))
		for p in pz:
			if p[2]==1 or p[1]==1:
				print (p)'''

	prf_test=precision_recall_fscore_support(res_test, labels_test, average='macro')
	prf_train=precision_recall_fscore_support(res_train, labels_train, average='macro')
	prf_detail_test=precision_recall_fscore_support(res_test, labels_test, average=None)
	prf_detail_train=precision_recall_fscore_support(res_train, labels_train, average=None)
	prf_test=[str(p) for p in prf_test]
	prf_test=' '.join(prf_test)
	prf_train=[str(p) for p in prf_train]
	prf_train=' '.join(prf_train)


	'''print("\n\n#%d Convolutional Testing Accuracy: %.2f%% Training Accuracy: %.2f%% Training Time: %.2fs Testing Time: %.2fs" % (run+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
	print("\nActual Testing Accuracy: %.2f%% Training Accuracy: %.2f%%" % (result_test2, result_train2))
	print("#Testing PRF: %s%%\nTraining PRF: %s%%" % (prf_test, prf_train))
	print("#Classwise Testing  & Training PRFS:\n")
	for clidx in range(len(oplabels)):
		print(oplabels[clidx]+": "+str(prf_detail_test[0][clidx])+" ; "+str(prf_detail_test[1][clidx])+" ; "+str(prf_detail_test[2][clidx])+" ; "+str(prf_detail_test[3][clidx])+" || "+str(prf_detail_train[0][clidx])+" ; "+str(prf_detail_train[1][clidx])+" ; "+str(prf_detail_train[2][clidx])+" ; "+str(prf_detail_train[3][clidx])+'\n')
	'''
	print("\nActual Testing Accuracy: %.2f%% Training Accuracy: %.2f%%" % (result_test2, result_train2))
	acc.append(result_test2)
	
print('Max Acc:', max(acc))
print('Min Acc:', min(acc))
print('Avg Acc:', sum(acc)/len(acc))

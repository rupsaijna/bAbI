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

context_length=2
qr=['query_word']
gram_base=["pos_", "tag_", "ent_type_", "is_alpha", "is_stop", "is_digit", "is_lower", "is_upper","is_punct", "is_left_punct", "is_right_punct", "is_bracket", "is_quote", "dep_", "head.pos_", "head.head.pos_"]
gram_base+=qr
addendum_context=['_wb'+str(l) for l in range(context_length,0,-1)]+['_wt']+['_wa'+str(l) for l in range(1,context_length+1)]
exf=['text','word_idx','label']

CLAUSES=1000
T=1500
s=27.0
weighting = True
motif_length=3


def find_uniques_length(df, ignoreheaders):
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
	temp_cols=np.array(temp_cols)
	print(temp_cols.shape)
	t=temp_cols.reshape(len(addendum_context),1,len(gram_base))
	print(t.shape)
	print(t)
	return newX	

glove_features_train=pd.read_pickle('../../pickles/spacy/nonbinarized_features_context2_train_glove.pkl')
grammar_features_train=pd.read_pickle('../../pickles/spacy/nonbinarized_features_context2_train_gram.pkl')

glove_features_test=pd.read_pickle('../../pickles/spacy/nonbinarized_features_context2_test_glove.pkl')
grammar_features_test=pd.read_pickle('../../pickles/spacy/nonbinarized_features_context2_test_gram.pkl')

'''
a=set(grammar_features_train.columns.tolist())
b=set(grammar_features_test.columns.tolist())
gram_headers=list(a.intersection(b))'''

assert(grammar_features_train['label'].tolist()==glove_features_train['label'].tolist())
assert(grammar_features_test['label'].tolist()==glove_features_test['label'].tolist())

labels_train=grammar_features_train['label']
labels_test=grammar_features_test['label']


print('gram',grammar_features_train.shape, len(labels_train))
print('glove',glove_features_train.shape, len(labels_train))
				 
print('gram',grammar_features_test.shape, len(labels_test))
print('glove',glove_features_test.shape, len(labels_test))

#########Grammar##########
print("Grammar")

combo_train=grammar_features_train
combo_test=grammar_features_test
				 
#combo_train= pd.concat([grammar_features_train, glove_features_train], axis=1, join='inner')
#combo_test= pd.concat([grammar_features_test, glove_features_test], axis=1, join='inner')
#combo_train = combo_train.loc[:,~combo_train.columns.duplicated()]
#combo_test = combo_test.loc[:,~combo_test.columns.duplicated()]
#combo_train=combo_train.drop(columns=[ 'info2_wa2'])
#combo_test=combo_test.drop(columns=[ 'info2_wa2'])

remheaders=['text','label', 'word_idx']

a=set(combo_train.columns.tolist())
b=set(combo_test.columns.tolist())
combo_headers=list(a.intersection(b))

print('combo train',combo_train.shape)
print('combo test',combo_test.shape)

				 
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
				 
'''# Setup
tm = MultiClassConvolutionalTsetlinMachine2D(CLAUSES, T, s, (motif_length, 1), weighted_clauses=weighting)

# Training
for i in range(5):
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
	print("\n#Testing PRF: %s%%\nTraining PRF: %s%%" % (prf_test, prf_train))'''

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

X_train2 = combo_train.reshape((combo_train.shape[0],5,1,int(combo_train.shape[1]/5)))
X_test2 = combo_test.reshape((combo_test.shape[0],5,1,int(combo_train.shape[1]/5)))

print('reshaped train',X_train2.shape)
print('reshaped test',X_test2.shape)
				 
'''# Setup
tm = MultiClassConvolutionalTsetlinMachine2D(CLAUSES, T, s, (motif_length, 1), weighted_clauses=weighting)

# Training
for i in range(5):
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
	print("\n#Testing PRF: %s%%\nTraining PRF: %s%%" % (prf_test, prf_train))'''

	
#########Combo##########
print("\n\nCombo")

X_train3 = np.concatenate((X_train,X_train2), axis=3)
X_test3 = np.concatenate((X_test,X_test2), axis=3)

print('reshaped train',X_train3.shape)
print('reshaped test',X_test3.shape)
				 
# Setup
tm = MultiClassConvolutionalTsetlinMachine2D(CLAUSES, T, s, (motif_length, 1), weighted_clauses=weighting)

# Training
for i in range(5):
	start_training = time()
	tm.fit(X_train3, labels_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	res_test=tm.predict(X_test3)
	res_train=tm.predict(X_train3) 
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
	
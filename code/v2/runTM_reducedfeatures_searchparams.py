import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sys
import os
import time
sys.path.append('../../pyTsetlinMachineParallel/')
#from tm import MultiClassConvolutionalTsetlinMachine2D
from tm import MultiClassTsetlinMachine
import numpy as np
import re
import string
import pickle
import csv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt


oplabels=['0','1']
context_length=4
qr=['query_word']
gram_base=["pos_", "tag_", "ent_type_", "is_alpha", "is_stop", "is_digit", "is_lower", "is_upper","is_punct", "is_left_punct", "is_right_punct", "is_bracket", "is_quote", "dep_", "head.pos_", "head.head.pos_"]
gram_base+=qr
addendum_context=['_wb'+str(l) for l in range(context_length,0,-1)]+['_wt']+['_wa'+str(l) for l in range(1,context_length+1)]
exf=['text','word_idx','label']

CLAUSES=160
T=90
s=2.5
weighting = True
motif_length=7
training_epoch=35
RUNS=100


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

def numerize(df, list_uniques, list_columns):
	temp_cols=[]
	newX=np.zeros((df.shape[0], len(gram_base)*len(addendum_context)), dtype=np.int32)
	startind=0
	for contextid in addendum_context:
		for colname_base in gram_base:
			colname=colname_base+contextid
			ul=list(list_uniques[gram_base.index(colname_base)])
			arr=df[colname].tolist()
			tempx=[ul.index(arr[pos]) for pos in range(len(arr))]				   
			temp_cols.append(colname)
			newX[:,startind]=tempx
			startind+=1
	'''temp_cols=np.array(temp_cols)
	print(temp_cols.shape)
	t=temp_cols.reshape(len(addendum_context),1,len(gram_base))
	print(t.shape)
	print(t)'''
	return newX	

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

def binarize_selected(df, list_uniques, list_columns):
	temp_cols=[]
	sum_size=np.sum([len(s) for s in list_uniques])
	newX=np.zeros((df.shape[0], sum_size*len(addendum_context)), dtype=np.int32)
	startind=0
	for contextid in addendum_context:
		for colname_base in gram_base:
			colname=colname_base+contextid
			if colname in list_columns:
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
	
	newX=newX[:,:startind]
	return newX	


glove_features_train=pd.read_pickle('../../pickles/spacy/nonbinarized_features_context'+str(context_length)+'_train_glove.pkl')
grammar_features_train=pd.read_pickle('../../pickles/spacy/nonbinarized_features_context'+str(context_length)+'_train_gram.pkl')

glove_features_test=pd.read_pickle('../../pickles/spacy/nonbinarized_features_context'+str(context_length)+'_test_glove.pkl')
grammar_features_test=pd.read_pickle('../../pickles/spacy/nonbinarized_features_context'+str(context_length)+'_test_gram.pkl')

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

remheaders=['text','label', 'word_idx']

print('combo train',combo_train.shape)
print('combo test',combo_test.shape)

				 
list_of_uniques=find_uniques_length(combo_train, remheaders)
#print(list_of_uniques)
#print('len col',len(baselist_gram))

usum=np.sum([len(s) for s in list_of_uniques])
print('sum', usum)	

combo_train_num=numerize(combo_train, list_of_uniques, gram_base)
combo_test_num=binarize(combo_test, list_of_uniques, gram_base)

print(combo_train_num.shape)
print(combo_test_num.shape)

colnames=list(combo_train.columns)
colnames=[c for c in colnames if c not in remheaders ]

fout=open('paramsearch_runTM_reducedfeatures.txt','w')
fout.write('Features\tCLAUSES\tT\ts\tMax_Tr\tMaxTs\tAvg_Tr\tAvgTs\tFeatures\n')
fout.close()

labels_test_indx=np.where(labels_test==1)
labels_train_indx=np.where(labels_train==1)

for NUM_FEATURES in range(3, len(colnames),5):
	SKB = SelectKBest(chi2, k=NUM_FEATURES)
	SKB.fit(combo_train_num, labels_train)
	selected_features = SKB.get_support(indices=True)
	cs=[colnames[sf] for sf in selected_features]

	Xtrain=binarize_selected(combo_train, list_of_uniques, cs)
	Xtest=binarize_selected(combo_test, list_of_uniques, cs)

	# Setup
	for CLAUSES in range(10,500, 10):
		for T in range(5, CLAUSES*2, 5):
			for s in np.arange(0.2, 40, 0.5):
				tm = MultiClassTsetlinMachine(CLAUSES, T, s, weighted_clauses=weighting)
				acc=[]
				acc_train=[]
				# Training
				for i in range(RUNS):
					print(i)
					start_training = time()
					tm.fit(Xtrain, labels_train, epochs=50, incremental=True)
					stop_training = time()

					start_testing = time()
					res_test=tm.predict(Xtest)
					res_train=tm.predict(Xtrain) 

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

					acc.append(result_test2)
					acc_train.append(result_train2)

				fout=open('paramsearch_runTM_reducedfeatures.txt','a')
				fout.write(str(NUM_FEATURES)+'\t'+str(CLAUSES)+'\t'+str(T)+'\t'+str(s)+'\t'+str(round(max(acc_train),2))+'\t'+str(round(max(acc),2))+'\t'+str(round(sum(acc_train)/len(acc_train),2))+'\t'+str(round(sum(acc)/len(acc),2))+'\t'+(',').join(cs)+'\n')
				fout.close()
'''
plt.plot(np.arange(1,len(acc)+1),acc, label = "Test")
plt.plot(np.arange(1,len(acc)+1),acc_train, label = "Test")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig('accuracy.png')
'''
'''
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


	prf_test=precision_recall_fscore_support(res_test, labels_test, average='macro')
	prf_train=precision_recall_fscore_support(res_train, labels_train, average='macro')
	prf_detail_test=precision_recall_fscore_support(res_test, labels_test, average=None)
	prf_detail_train=precision_recall_fscore_support(res_train, labels_train, average=None)
	prf_test=[str(p) for p in prf_test]
	prf_test=' '.join(prf_test)
	prf_train=[str(p) for p in prf_train]
	prf_train=' '.join(prf_train)

	print("\nActual Testing Accuracy: %.2f%% Training Accuracy: %.2f%%" % (result_test2, result_train2))
	acc.append(result_test2)
	
print('Max Acc:', max(acc))
print('Min Acc:', min(acc))
print('Avg Acc:', sum(acc)/len(acc))
'''	
	

import sys
sys.path.append('../pyTsetlinMachineParallel/')
from tm import MultiClassTsetlinMachine
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
import random


def get_random_local_view(X_test, num, local_clause_file, tm, clause_file):
	print("Going Local")
	indx=random.randint(num,len(X_test))
	print(X_test[:indx])
	temp_X_test=X_test[indx-num:indx]
	temp_y_test=y_test[indx-num:indx]
	temp_X_test_sent=[]
	for l in range(len(temp_X_test)):
		temp_sent=[]
		line=temp_X_test[l]
		for ft in range(len(line)):
			if line[ft]==1:
				temp_sent.append(featureheaderset[ft])
		temp_X_test_sent.append(' '.join(temp_sent))
		print(temp_sent, temp_y_test[l], labels_set[temp_y_test[l]])

	if os.path.exists(local_clause_file):
		print('overwirting previous '+local_clause_file)
		os.remove(local_clause_file)
	fo=open(local_clause_file,'w')
	fo.write('Example Class Clause Cl.Val\n')
	fo.close()
	res=tm.predict_and_printlocal(temp_X_test, local_clause_file)

	print('Result:',res)

	local_clauses=pd.read_csv(local_clause_file,sep=' ')
	for ts in range(len(temp_X_test_sent)):
		for ind,row in local_clauses.iterrows():
			if row['Example']==ts:
				local_clauses.loc[local_clauses.index[ind], 'Example_BoW']=temp_X_test_sent[ts]
				local_clauses.loc[local_clauses.index[ind], 'ClassName']=labels_set[int(row['Class'])]
	all_clauses=pd.read_csv(clause_file,sep='\t')
	for ind,row in local_clauses.iterrows():
		classname=row['ClassName']
		clauseid=int(row['Clause'])
		clausetext=all_clauses[(all_clauses['ClauseNum']==clauseid) & (all_clauses['class']==classname) ]['Clause'].values
		local_clauses.loc[local_clauses.index[ind], 'ClauseText']=clausetext
		star=''
		if row['Class']==temp_y_test[row['Example']]:
			star+='G'
		if row['Class']==temp_y_test[row['Example']]:
			star+='P'
		local_clauses.loc[local_clauses.index[ind], 'CorrectLabel']=star

	local_clauses=local_clauses.sort_values(by=['Example', 'Class'])

	local_clauses.to_csv(local_clause_file, sep='\t', index=False)
	print('Local Clauses written to:'+local_clause_file)

def wrote_clauses(clause_file, X, tm, featureheaderset, labels_set):
	fout_c=open(clause_file,'w')
	NUM_FEATURES=X.shape[1]
	fout_c.write('ClauseNum\tClause\tp/n\tclass\n')
	feature_vector=np.zeros(NUM_FEATURES*2)
	for cur_cls in range(len(labels_set)):
		for cur_clause in range(CLAUSES):
			if cur_clause%2==0:
				clause_type='positive'
			else:
				clause_type='negative'
			this_clause=str(cur_clause)+'\t'
			for f in range(0,NUM_FEATURES):
				action_plain = tm.ta_action(int(cur_cls), cur_clause, f)
				action_negated = tm.ta_action(int(cur_cls), cur_clause, f+NUM_FEATURES)
				feature_vector[f]=action_plain
				feature_vector[f+NUM_FEATURES]=action_negated
				if action_plain==1:
					this_clause+=featureheaderset[f]+';'
				#if action_negated==1:
				#	this_clause+='#'+featureheaderset[f]+';'
			this_clause+='\t'+clause_type+'\t'+str(labels_set[cur_cls])	
			fout_c.write(str(this_clause)+'\n')
	fout_c.close()

	print('Clauses written at :'+ clause_file)
	
fname=sys.argv[1]
local_clause_file='local_clauses_rel.csv'

clause_file=fname.replace('.txt','_rel_clauses.txt')

#df 
dfname=fname.replace('.txt','')+'_relationalfeatureset.pkl'

df_transformed= pd.read_pickle(dfname)
#df_transformed = pd.read_pickle(fname.replace('.txt','')+'_transformed_relationalfeatureset.pkl')


featureset_transformed_y = df_transformed['ANSWER'].values

labels_set=list(set(featureset_transformed_y))

if 'transformed' not in dfname:
	featureset_transformed_y=[labels_set.index(ft) for ft in featureset_transformed_y]

df_transformed_X= df_transformed.drop(columns=['ANSWER'])
featureheaderset=df_transformed_X.columns
featureheaderset=[f.replace(';','_') for f in featureheaderset]
featureset_transformed_X = df_transformed_X.to_numpy()

'''
rel_classify : _transformed_relationalfeatureset.pkl
CLAUSES=25
T=30
s=3#s=10
weighting = True
training_epoch=5
RUNS=100
'''

CLAUSES=250
T=60
s=37
weighting = True
training_epoch=5
RUNS=100

X_train, X_test, y_train, y_test = train_test_split(featureset_transformed_X, featureset_transformed_y, test_size=0.30, random_state=42, shuffle=True)

tm = MultiClassTsetlinMachine(CLAUSES, T, s, weighted_clauses=weighting,append_negated=False)

allacc=[]
for i in range(RUNS):
	tm.fit(X_train, y_train, epochs=training_epoch, incremental=True)
	res_test=tm.predict(X_test)
	print(res_test)
	acc_test = 100*(res_test == y_test).mean()

	allacc.append(acc_test)
	
	#print(type(y_test), type(res_test), len(y_test), len(res_test))
	prf_test_macro=precision_recall_fscore_support(list(res_test), list(y_test), average='macro')
	

	prf_test_macro=[str(round(p,2)) for p in prf_test_macro[:-1]]
	
	prf_test_micro=precision_recall_fscore_support(list(res_test), list(y_test), average='micro')
	prf_test_micro=[str(round(p,2)) for p in prf_test_micro[:-1]]
	
	prf_test_class=precision_recall_fscore_support(list(res_test), list(y_test), average=None)
	
	print("\n\n#%d Testing Accuracy: %.2f%% " % (i+1, acc_test))
	#print("\n#Testing PRF Macro: " + ', '.join(prf_test_macro))
	#print("\nTesting PRF Micro: " + ', '.join(prf_test_micro))
	for ls in range(len(labels_set)):
		print(str(labels_set[ls])+' : '+str(round(prf_test_class[0][ls],2))+', '+str(round(prf_test_class[1][ls],2))+', '+str(round(prf_test_class[2][ls],2))+', '+str(round(prf_test_class[3][ls],2)))

print('Over '+str(RUNS)+' runs: '+str(np.mean(allacc, axis=0))+' +/- '+str(np.std(allacc, axis=0)))

lastruns=int(RUNS/3)
print('Last '+str(lastruns)+' runs: '+str(np.mean(allacc[-lastruns:], axis=0))+' +/- '+str(np.std(allacc[-lastruns:], axis=0)))


####ALL CLAUSES###
wrote_clauses(clause_file, df_transformed_X, tm, featureheaderset, labels_set)

####LOCAL VIEW###
get_random_local_view(X_test, 20, local_clause_file, tm, clause_file)

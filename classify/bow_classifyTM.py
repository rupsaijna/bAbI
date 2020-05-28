import sys
sys.path.append('../pyTsetlinMachineParallel/')
from tm import MultiClassTsetlinMachine
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd

#fname='../generated/generated2.txt'
fname=sys.argv[1]

clause_file=fname.replace('.txt','_clauses.txt')

f=open(fname.replace('_sentenceleveltransform','').replace('.txt','_meta.txt'),'r')
lines=f.readlines()

for l in lines[3:]:
	lt=l.split('\t')
	if len(lt)>1 and lt[0]==fname:
		labels_set=lt[1].split(',')
		featureheaderset=lt[2].replace('\n','').split(',')

if 'sentenceleveltransform' not in fname:
	CLAUSES=22
	T=19
	s=9.5
	weighting = True
	training_epoch=5
	RUNS=50

else:
	CLAUSES=2
	T=4
	s=1
	weighting = True
	training_epoch=1
	RUNS=100
	
featureset=np.load(fname.replace('.txt','')+'_featureset.npy')

X=featureset[:,:-1]
y=featureset[:,-1]
y=[int(yy) for yy in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)

tm = MultiClassTsetlinMachine(CLAUSES, T, s, weighted_clauses=weighting,append_negated=True)

allacc=[]
for i in range(RUNS):
	tm.fit(X_train, y_train, epochs=training_epoch, incremental=True)
	res_test=tm.predict(X_test)
	
	acc_test = 100*(res_test == y_test).mean()

	allacc.append(acc_test)
	prf_test_macro=precision_recall_fscore_support(res_test, y_test, average='macro')
	prf_test_macro=[str(round(p,2)) for p in prf_test_macro[:-1]]
	
	prf_test_micro=precision_recall_fscore_support(res_test, y_test, average='micro')
	prf_test_micro=[str(round(p,2)) for p in prf_test_micro[:-1]]
	
	prf_test_class=precision_recall_fscore_support(res_test, y_test, average=None)
	
	print("\n\n#%d Testing Accuracy: %.2f%% " % (i+1, acc_test))
	#print("\n#Testing PRF Macro: " + ', '.join(prf_test_macro))
	#print("\nTesting PRF Micro: " + ', '.join(prf_test_micro))
	for ls in range(len(labels_set)):
		print(labels_set[ls]+' : '+str(round(prf_test_class[0][ls],2))+', '+str(round(prf_test_class[1][ls],2))+', '+str(round(prf_test_class[2][ls],2))+', '+str(round(prf_test_class[3][ls],2)))

print('Over '+str(RUNS)+' runs: '+str(np.mean(allacc, axis=0))+' +/- '+str(np.std(allacc, axis=0)))

lastruns=int(RUNS/3)
print('Last '+str(lastruns)+' runs: '+str(np.mean(allacc[-lastruns:], axis=0))+' +/- '+str(np.std(allacc[-lastruns:], axis=0)))

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
			if action_negated==1:
				this_clause+='#'+featureheaderset[f]+';'
		this_clause+='\t'+clause_type+'\t'+str(labels_set[cur_cls])	
		fout_c.write(str(this_clause)+'\n')
fout_c.close()

print('Clauses written at :'+ clause_file)

print(X_test[:2])
temp_X_test=X_test[:2]
temp_y_test=y_test[:2]
temp_X_test_sent=[]
for l in range(len(temp_X_test)):
	temp_sent=[]
	line=temp_X_test[l]
	for ft in range(len(line)):
		if line[ft]==1:
			temp_sent.append(featureheaderset[ft])
	temp_X_test_sent.append(temp_sent)
	print(temp_sent, temp_y_test[l], labels_set[temp_y_test[l]])

print(temp_X_test_sent)	
if os.path.exists('local_clauses.csv'):
    os.remove('local_clauses.csv')
fo=open('local_clauses.csv','w')
fo.write('Example Class CLause Cl.Val\n')
fo.close()
res=tm.predict_and_printlocal(temp_X_test, 'local_clauses.csv')

print('Result:',res)

local_clauses=pd.read_csv('local_clauses.csv',sep=' ')
print(local_clauses)
for ts in range(len(temp_X_test_sent)):
	for ind,row in local_clauses.iterrows():
		if row['Example']==ts:
			local_clauses.ix[ind,'ex_bow']=temp_X_test_sent[ts]
			
print(local_clauses)



import sys
sys.path.append('../pyTsetlinMachineParallel/')
from tm import MultiClassTsetlinMachine
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np

#fname='../generated/generated2.txt'
fname=sys.argv[1]
CLAUSES=40
T=25
s=2.5
weighting = True
training_epoch=1
RUNS=100

f=open(fname.replace('../generated/','../generated/meta_'),'r')
labels_set=f.readlines()[1].replace('\n','').split(',')
f.close()
labels_set=[ls.replace('the ','') for ls in labels_set]

featureset=np.load(fname.replace('.txt','')+'_featureset.npy')

X=featureset[:,:-1]
y=featureset[:,-1]
y=[int(yy) for yy in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)

tm = MultiClassTsetlinMachine(CLAUSES, T, s, weighted_clauses=weighting)

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

print(np.mean(allacc, axis=0),' +/- ',np.std(allacc, axis=0))

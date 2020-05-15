import sys
sys.path.append('../../pyTsetlinMachineParallel/')
from tm import MultiClassTsetlinMachine
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

fname='../generated/generated1.txt'
CLAUSES=40
T=30
s=2.5
weighting = True
training_epoch=1
RUNS=10

featureset=np.load(fname.replace('.txt','')+'_featureset.npy')

X=featureset[:,:-1]
y=featureset[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)

tm = MultiClassTsetlinMachine(CLAUSES, T, s, weighted_clauses=weighting)


for i in range(RUNS):
	print(i)
	tm.fit(X_train, y_train, epochs=training_epoch, incremental=True)
	res_test=tm.predict(X_test)
	
	acc_test = 100*(res_test == y_test).mean()
	
	prf_test_macro=precision_recall_fscore_support(res_test, labels_test, average='macro')
	prf_test_macro=[str(round(p,2)) for p in prf_test_macro]
	
	prf_test_micro=precision_recall_fscore_support(res_test, labels_test, average='micro')
	prf_test_micro=[str(round(p,2)) for p in prf_test_micro]
	
	print("\n\n#%d Testing Accuracy: %.2f%% " % (i+1, result_test))
	print("\n#Testing PRF Macro: " + ', '.join(prf_test_macro))
	print("\nTesting PRF Micro: " + ', '.join(prf_test_macro))


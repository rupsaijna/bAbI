sys.path.append('../../pyTsetlinMachineParallel/')
from tm import MultiClassTsetlinMachine


fname='../generated/generated1.txt'
CLAUSES=40
T=30
s=2.5
weighting = True
training_epoch=1
RUNS=10

featureset=np.load(fname.replace('.txt','')+'_featureset.npy')

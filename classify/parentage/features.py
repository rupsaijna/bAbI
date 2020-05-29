import numpy as np
import sys

#fname='../generated/generated2.txt'
fname=sys.argv[1]
f=open(fname,'r')
data=f.readlines()
f.close()

f=open(fname.replace('_sentenceleveltransform','').replace('.txt','_meta.txt'),'r')
labels_set=f.readlines()[1].replace('\n','').split(',')
f.close()

data=[d.replace('\n','') for d in data]

data=[d.split('\t') for d in data]

text=[d[0] for d in data]
labels=[int(d[1]) for d in data]
print(list(labels).count(0), list(labels).count(1), list(labels).count(2))
dcA
#labels_set=list(set(labels))

realtionships_set= set()
for t in text:
	sent=t.replace('\n','').split('.')
	for s in sent[:-1]:
		s=s.strip().replace('.','').split(' ')
		temp_rel=s[0]+'_'+s[-1]
		realtionships_set.add(temp_rel)
	

realtionships_set=list(realtionships_set)
numsentences=2

featureset=np.zeros((len(text),len(realtionships_set)+1))

print(realtionships_set)
print(labels_set)
print(set(labels))

featureheaders=[]
for n in realtionships_set:
	featureheaders.append('isParent('+str(n)+')')
textind=0
for t in text:
	sent=t.replace('\n','').split('.')
	temp_feature=np.zeros(len(realtionships_set)+1)
	for s in sent[:-1]:
		s=s.strip().replace('.','').split(' ')
		temp_rel=s[0]+'_'+s[-1]
		ind=realtionships_set.index(temp_rel)
		temp_feature[ind]=1
	temp_feature[-1]=labels[textind]
	featureset[textind]=temp_feature
	textind+=1
		
	
#print(featureset.shape, featureheaders, numsentences)	

np.save(fname.replace('.txt','')+'_featureset.npy', featureset)
f=open(fname.replace('_sentenceleveltransform','').replace('.txt','_meta.txt'),'a+')
f.write('\n'+fname+'\t'+','.join(labels_set)+'\t')
f.write(','.join(featureheaders)+'\n')
f.close()
#fs=np.load('featureset.npy')

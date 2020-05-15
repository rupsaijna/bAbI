import numpy as np

fname='../generated/generated1.txt'
f=open(fname,'r')
data=f.readlines()
f.close()

data=[d.replace('\n','') for d in data]

data=[d.split('\t') for d in data]

text=[d[0] for d in data]
labels=[d[1] for d in data]
labels_set=list(set(labels))

word_set_sentences= set()
word_set_questions= set()
newtext=[]
for t in text:
	sent=t.split('.')
	sent=[s.replace('.','').replace('?','').strip() for s in sent]
	sent=[s.split(' ') for s in sent]
	newtext.append(sent)
	tempsent=[word for subsent in sent[:-1] for word in subsent]
	for w in tempsent:
		word_set_sentences.add(w)
	for w in sent[-1]:
		word_set_questions.add(w)

word_set_sentences=list(word_set_sentences)
word_set_questions=list(word_set_questions)
numsentences=len(newtext[0])-1


featureset=np.zeros((len(newtext),len(word_set_sentences)*numsentences+len(word_set_questions)+1))

sentfeaturelen=len(word_set_sentences)
qsfeaturelen=len(word_set_questions)


print(word_set_sentences)
print(word_set_questions)
print(labels_set)

textind=0
for nt in newtext:
	startind=0
	for sentence in nt[:-1]:
		tempfeature=[1 if word_set_sentences[i] in sentence else 0 for i in range(sentfeaturelen)]
		featureset[textind,startind:startind+sentfeaturelen]=tempfeature
		startind=startind+sentfeaturelen
	tempfeature=[1 if word_set_questions[i] in nt[-1] else 0 for i in range(qsfeaturelen) ]
	featureset[textind,startind:-1]=tempfeature
	featureset[textind,-1]=labels_set.index(labels[textind])
	textind+=1
	
	
np.save(fname.replace('.txt','')+'_featureset.npy', featureset)

#fs=np.load('featureset.npy')
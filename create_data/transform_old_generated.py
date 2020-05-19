old_fname='../generated/generated1.txt'
f=open(old_fname,'r')

fout=open(old_fname.replace('.txt','_sentenceleveltransform.txt'),'w')

for line in f.readlines():
	parts=line.split('\t')
	sentences=parts[0].split('.')
	temp_sentences=''
	temp_label=''
	sentind=1
	qswords=sentences[-1].strip().split(' ')
	for sent in sentences[:-1]:
		words=sent.strip().split(' ')
		if words[0] in qswords[-1]:
			qswords[-1]=qswords[-1].replace(words[0],'PER'+str(sentind))
		words[0]='PER'+str(sentind)
		if words[-1] == parts[1]:
			temp_label='LOC'+str(sentind)
		words[-1]='LOC'+str(sentind)
		sentind+=1
		temp_sentences+=' '.join(words).strip()+'. '
	
	temp_sentences+=' '.join(qswords).strip()
	temp_sentences=temp_sentences.strip()
	temp_line=temp_sentences+'\t'+temp_label+'\t'+parts[2]
	fout.write(temp_line)


f.close()
fout.close()

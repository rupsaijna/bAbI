#X goes to a. Y goes to b. X goes to c. Where is X?

import random
from tqdm import tqdm
import pandas as pd

def print_numbers(num_sentences,len_names,len_locs,len_verbs,num_examples=0):
	print ("\nCurrently we can have:\nSentences per example:"+str(num_sentences)+"\n#Names:"+str(len_names)+"\n#Locations:"+str(len_locs)+"\n#Verbs:"+str(len_verbs))
	print ("\n#Total possible generated examples: "+str(len_names*len_locs*len_verbs*(len_names)*(len_locs-1)*(len_verbs)*num_sentences))
	if num_examples!=0:
		print ("\n#Examples to be generated:"+str(num_examples))
	print("##########################")


def confirm_numbers(num_sentences,len_names,len_locs,len_verbs):
	global names
	global locs
	global verbs
	while True:
		print_numbers(num_sentences,len_names,len_locs,len_verbs)
		num_examples=len_names*len_locs*len_verbs*(len_names)*(len_locs-1)*(len_verbs)*num_sentences
		
		cnt_names=len_names
		cnt_verbs=len_verbs
		cnt_locs=len_locs

		ans1 = input("\n>Do you want to limit number of names/locations/verbs? Y/N : ")
		
		if ans1.lower()=='y':
  
			cnt_names = input(str(len_names)+" names available. How many do you want to use? Type 0 to use all. ") 
			cnt_names=[int(cnt_names) if cnt_names!='0' else len_names][0]
			if(cnt_names<num_sentences):
				print('\n>>ERROR!!Must be greater than '+str(num_sentences)+',defaulting to original')
				cnt_names=len(names)
				continue

			cnt_locs = input(str(len_locs)+" locations available. How many do you want to use? Type 0 to use all. ") 
			cnt_locs=[int(cnt_locs) if cnt_locs!='0' else len_locs][0]
			if(cnt_locs<num_sentences):
				print('\n>>ERROR!!Must be greater than '+str(cnt_locs)+',defaulting to original')
				cnt_locs=len(locs)
				continue

			cnt_verbs = input(str(len_verbs)+" verbs available. How many do you want to use? Type 0 to use all. ") 
			cnt_verbs=[int(cnt_verbs) if cnt_verbs!='0' else len_verbs][0]

			print_numbers(num_sentences,cnt_names,cnt_locs,cnt_verbs)
			len_names=cnt_names
			len_locs=cnt_locs
			len_verbs=cnt_verbs
			num_examples=len_names*len_locs*len_verbs*(len_names)*(len_locs-1)*(len_verbs)*num_sentences
		
		ans2 = input("\n>Do you want to limit number of generated examples? Y/N : ")
		
		if ans2.lower()=='y':
			ne=int(input("How many examples?[1-"+str(num_examples)+"] "))
			if(ne>num_examples):
				print('\n>>ERROR!!Must be lesser than '+str(num_examples)+',defaulting to original')
				len_locs=len(locs)
				len_names=len(names)
				len_verbs=len(verbs)
				continue
			else:
				num_examples=ne
			
		print_numbers(num_sentences,cnt_names,cnt_locs,cnt_verbs,num_examples)	
		ans3 = input("\n>Proceed to generate? Y/N : ")
		
		if ans3.lower()=='y':
			print("Proceeding....")
			return len_names,len_locs,len_verbs,num_examples
		else:
			len_locs=len(locs)
			len_names=len(names)
			len_verbs=len(verbs)


def generate(num_sentences,names,locs,verbs,num_examples):
	pbar = tqdm(total = num_examples)
	examples=[]
	examples_transformed=[]
	features=[]
	all_possible_combinations=[n+';'+str(l) for n in names for l in locs]
	all_possible_combinations+=['q_'+n for n in names]
	all_possible_combinations+=['ANSWER']
	df=pd.DataFrame(columns=all_possible_combinations)
	transformed_names=['PER'+str(ix) for ix in range(num_sentences)]
	transformed_locs=['LOC'+str(ix) for ix in range(num_sentences)]
	all_possible_combinations_transformed=[n+';'+str(l) for n in transformed_names for l in transformed_locs]
	all_possible_combinations_transformed+=['q_'+n for n in transformed_names]
	all_possible_combinations_transformed+=['ANSWER']
	df_transformed=pd.DataFrame(columns=all_possible_combinations_transformed)
	while len(examples) < num_examples:
		temp_names=random.sample(names, k=num_sentences) #without repeatation
		temp_locs=random.sample(locs, k=num_sentences) #without repeatation
		temp_verbs=random.choices(verbs, k=num_sentences) #with repeatation
		temp_example=''
		temp_example_transformed=''
		temp_answers={tn:'' for tn in temp_names}
		temp_names_used=[]
		temp_featured=[]
		df = df.append(pd.Series(0, index=df.columns), ignore_index=True) ##relation
		df_transformed = df_transformed.append(pd.Series(0, index=df_transformed.columns), ignore_index=True) ##relation
		for i in range(num_sentences):
			tempindx=random.randint(0,num_sentences-1)
			temp_sentence=temp_names[tempindx]+' '+temp_verbs[i]+' '+temp_locs[i]+'. '
			temp_answers[temp_names[tempindx]]=temp_locs[i]
			temp_names_used.append(tempindx)
			df.loc[df.index[-1],temp_names[tempindx]+';'+temp_locs[i]]=i+1 ##relation
			temp_sentence_transformed='PER'+str(tempindx)+' '+temp_verbs[i]+' '+'LOC'+str(i)+'. '
			temp_example+=temp_sentence
			temp_example_transformed+=temp_sentence_transformed
			df_transformed.loc[df.index[-1],'PER'+str(tempindx)+';'+'LOC'+str(i)]=i+1 ##relation
		qs_num=random.randint(0,num_sentences-1)
		temp_qs='Where is '+temp_names[temp_names_used[qs_num]]+'?'
		temp_qs_transformed='Where is '+'PER'+str(temp_names_used[qs_num])+'?'
		temp_example+=temp_qs
		temp_example_transformed+=temp_qs_transformed
		
		df.loc[df.index[-1],'q_'+temp_names[temp_names_used[qs_num]]]=1 ##relation
		df_transformed.loc[df.index[-1],'q_'+'PER'+str(temp_names_used[qs_num])]=1 ##relation

		temp_ans=temp_answers[temp_names[temp_names_used[qs_num]]].replace('the ','')+'\t'+str(qs_num+1)
		temp_ans_transformed='LOC'+str(temp_locs.index(temp_answers[temp_names[temp_names_used[qs_num]]]))+'\t'+str(qs_num+1)

		df.loc[df.index[-1],'ANSWER']=temp_answers[temp_names[temp_names_used[qs_num]]].replace('the ','') ##relation
		df_transformed.loc[df.index[-1],'ANSWER']=int(temp_locs.index(temp_answers[temp_names[temp_names_used[qs_num]]])) ##relation
		temp_example+='\t'+temp_ans
		temp_example_transformed+='\t'+temp_ans_transformed
		if temp_example not in examples:
			examples.append(temp_example)
			examples_transformed.append(temp_example_transformed)
			pbar.update(1)
	pbar.close()		
	return examples, examples_transformed,df, df_transformed
	
######################################################################				
			
with open('names.txt') as f:
    names = f.readlines()
    
names=[n.replace('\n','') for n in names]
random.shuffle(names)

with open('locations.txt') as f:
    locs = f.readlines()

locs=[l.replace('\n','') for l in locs]
random.shuffle(locs)

    
with open('verbs_past.txt') as f:
    verbs = f.readlines()  
    
verbs=[v.replace('\n','') for v in verbs]
random.shuffle(verbs)

##input
opfileans=input("\n>Output file is: generated_data.txt Y/N? ")
if opfileans.lower()=='y':
	opfile='generated_data.txt'
else:
	opfile=str(input("\n>Enter output file name (.txt only): "))

num_sentences=input("\n>How many sentences per example? e.g. 2 ") 
num_sentences=int(num_sentences)
##confirm numbers to be generated
len_names,len_locs,len_verbs,num_examples = confirm_numbers(num_sentences,len(names),len(locs),len(verbs))

names=names[:len_names]
locs=locs[:len_locs]
verbs=verbs[:len_verbs]

examples, examples_transformed, df, df_transformed=generate(num_sentences,names,locs,verbs,num_examples)
fo=open(opfile,'w')
for e in examples:
	fo.write(e+'\n')
fo.close()

opfile2=opfile.replace('.txt','_sentenceleveltransform.txt')
fo=open(opfile2,'w')
for e in examples_transformed:
	fo.write(e+'\n')
fo.close()
	
opfile_meta=opfile.replace('.txt','_meta.txt')	
fo=open(opfile_meta,'w')	
fo.write(','.join(names)+'\n')
fo.write(','.join(locs)+'\n')
fo.write(','.join(verbs)+'\n')

fo.close()

opfile_df=opfile.replace('.txt','_df.pkl')
df.to_pickle(opfile_df)


opfile_df_transformed=opfile.replace('.txt','_df_transformed.pkl')
df_transformed.to_pickle(opfile_df_transformed)

print('Output written to '+opfile)

import random
from tqdm import tqdm

relations=['Grandparent','Sibling','Unrelated']

def print_numbers(num_sentences,len_names,num_examples=0):
	print ("\nCurrently we can have:\nSentences per example:"+str(num_sentences)+"\n#Names:"+str(len_names))
	print ("\n#Total possible generated examples: "+str((2*len_names*(len_names-1)*(len_names-2))+(len_names*(len_names-1)*(len_names-2)*(len_names-3))))
	if num_examples!=0:
		print ("\n#Examples to be generated:"+str(num_examples))
	print("##########################")


def confirm_numbers(num_sentences,len_names):
	global names
	while True:
		print_numbers(num_sentences,len_names)
		num_examples=(2*len_names*(len_names-1)*(len_names-2))+(len_names*(len_names-1)*(len_names-2)*(len_names-3))
		
		cnt_names=len_names

		ans1 = input("\n>Do you want to limit number of names? Y/N : ")
		
		if ans1.lower()=='y':
  
			cnt_names = input(str(len_names)+" names available. How many do you want to use? Type 0 to use all. ") 
			cnt_names=[int(cnt_names) if cnt_names!='0' else len_names][0]
			if(cnt_names<num_sentences):
				print('\n>>ERROR!!Must be greater than '+str(num_sentences)+',defaulting to original')
				cnt_names=len(names)
				continue

			print_numbers(num_sentences,cnt_names)
			len_names=cnt_names

			num_examples=(2*len_names*(len_names-1)*(len_names-2))+(len_names*(len_names-1)*(len_names-2)*(len_names-3))
		
		ans2 = input("\n>Do you want to limit number of generated examples? Y/N : ")
		
		if ans2.lower()=='y':
			ne=int(input("How many examples?[1-"+str(num_examples)+"] "))
			if(ne>num_examples):
				print('\n>>ERROR!!Must be lesser than '+str(num_examples)+',defaulting to original')
				len_names=len(names)
				continue
			else:
				num_examples=ne
			
		print_numbers(num_sentences,cnt_names,num_examples)	
		ans3 = input("\n>Proceed to generate? Y/N : ")
		
		if ans3.lower()=='y':
			print("Proceeding....")
			return len_names,num_examples
		else:
			len_names=len(names)


def generate(num_sentences,names,num_examples):
	fo=open(opfile,'w')
	temp_names=names.copy()
	pbar = tqdm(total = num_examples)
	examples=[]
	examples_transformed=[]
	cnt_0=0
	cnt_1=0
	cnt_2=0
	
	while cnt_2 < int(num_examples/3) and cnt_1 < int(num_examples/3):
		#temp_names=random.sample(names, k=num_sentences) #without repeatation
		#temp_locs=random.sample(locs, k=num_sentences) #without repeatation
		#temp_verbs=random.choices(verbs, k=num_sentences) #with repeatation
		temp_example_fr=''
		temp_example_transformed_fr=''
		
		###
		temp_names=names.copy()
		#print(temp_names,names, len(examples))
		temp_names_first=random.sample(temp_names, k=2)
		temp_example_fr=temp_names_first[0]+' is the parent of '+temp_names_first[1]+'. '
		temp_example_transformed_fr='X is the parent of Y. '
		temp_names.remove(temp_names_first[0])
		temp_names.remove(temp_names_first[1])
		for rel in [0,1,2]:
			print(rel)
			if rel==0:
				temp_names_second=random.sample(temp_names, k=1)
				temp_example=temp_example_fr+temp_names_first[1]+' is the parent of '+temp_names_second[0]+'. '
				temp_example+='How are '+temp_names_first[0]+' and '+temp_names_second[0]+' related?'
				temp_example_transformed=temp_example_transformed_fr+'Y is the parent of Z. '
				temp_example_transformed+='How are X and Z related?'
				
				temp_example+='\t'+str(0)
				temp_example_transformed+='\t'+str(0)
				if temp_example not in examples:
					cnt_0+=1
					fo.write(temp_example+'\n')
					examples.append(temp_example)
					examples_transformed.append(temp_example_transformed)
					pbar.update(1)
			if rel==1:
				temp_names_second=random.sample(temp_names, k=1)
				temp_example=temp_example_fr+temp_names_first[0]+' is the parent of '+temp_names_second[0]+'. '
				temp_example+='How are '+temp_names_first[0]+' and '+temp_names_second[0]+' related?'
				temp_example_transformed=temp_example_transformed_fr+'X is the parent of Z. '
				temp_example_transformed+='How are X and Z related?'
				temp_example+='\t'+str(1)
				temp_example_transformed+='\t'+str(1)
				if temp_example not in examples:
					cnt_1+=1
					fo.write(temp_example+'\n')
					examples.append(temp_example)
					examples_transformed.append(temp_example_transformed)
					pbar.update(1)
			if rel==2:
				temp_names_second=random.sample(temp_names, k=2)
				temp_example=temp_example_fr+temp_names_second[0]+' is the parent of '+temp_names_second[1]+'. '
				temp_example+='How are '+temp_names_first[0]+' and '+temp_names_second[1]+' related?'
				temp_example_transformed=temp_example_transformed_fr+'A is the parent of Z. '
				temp_example_transformed+='How are X and Z related?'
				temp_example+='\t'+str(2)
				temp_example_transformed+='\t'+str(2)
				if temp_example not in examples:
					cnt_2+=1
					fo.write(temp_example+'\n')
					examples.append(temp_example)
					examples_transformed.append(temp_example_transformed)
					pbar.update(1)
		
			
	pbar.close()	
	fo.close()
	print(cnt_0,cnt_1,cnt_2)
	return examples, examples_transformed
	
######################################################################				
			
with open('names.txt') as f:
    names = f.readlines()
    
names=[n.replace('\n','') for n in names]
random.shuffle(names)

##input
opfileans=input("\n>Output file is: '../../generated/generated_parentage_data.txt' Y/N? ")
if opfileans.lower()=='y':
	opfile='../../generated/generated_parentage_data.txt'
else:
	opfile=str(input("\n>Enter output file name (.txt only): "))

#num_sentences=input("\n>How many sentences per example? e.g. 2 ") 
#num_sentences=int(num_sentences)

num_sentences=2
##confirm numbers to be generated
len_names,num_examples = confirm_numbers(num_sentences,len(names))

names=names[:len_names]

examples, examples_transformed=generate(num_sentences,names,num_examples)

'''for e in examples:
	print(e)
	fo.write(e+'\n')
fo.close()'''

opfile2=opfile.replace('.txt','_sentenceleveltransform.txt')
fo=open(opfile2,'w')
for e in examples_transformed:
	fo.write(e+'\n')
fo.close()
	
opfile=opfile.replace('.txt','_meta.txt')	
fo=open(opfile,'w')	
fo.write(','.join(names)+'\n')
fo.write(','.join(relations)+'\n')
fo.close()
print('Output written to '+opfile)

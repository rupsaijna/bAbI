import random
from tqdm import tqdm

def print_numbers(num_sentences,len_names,len_locs,len_verbs,num_examples=0):
	print ("\nCurrently we can have:\nSentences per example:"+str(num_sentences)+"\n#Names:"+str(len_names)+"\n#Locations:"+str(len_locs)+"\n#Verbs:"+str(len_verbs))
	print ("\n#Total possible generated examples: "+str(len_names*len_locs*len_verbs*(len_names-1)*(len_locs-1)*(len_verbs)*num_sentences))
	if num_examples!=0:
		print ("\n#Examples to be generated:"+str(num_examples))
	print("##########################")


def confirm_numbers(num_sentences,len_names,len_locs,len_verbs):
	global names
	global locs
	global verbs
	while True:
		print_numbers(num_sentences,len_names,len_locs,len_verbs)
		num_examples=len_names*len_locs*len_verbs*(len_names-1)*(len_locs-1)*(len_verbs)*num_sentences
		
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
			num_examples=len_names*len_locs*len_verbs*(len_names-1)*(len_locs-1)*(len_verbs)*num_sentences
		
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
	while len(examples) < num_examples:
		temp_names=random.sample(names, k=num_sentences) #without repeatation
		temp_locs=random.sample(locs, k=num_sentences) #without repeatation
		temp_verbs=random.choices(verbs, k=num_sentences) #with repeatation
		temp_example=''
		for i in range(num_sentences):
			temp_sentence=temp_names[i]+' '+temp_verbs[i]+' '+temp_locs[i]+'. '
			temp_example+=temp_sentence
		qs_num=random.randint(0,num_sentences-1)
		temp_qs='Where is '+temp_names[qs_num]+'?'
		temp_example+=temp_qs
		temp_ans=temp_locs[qs_num].replace('the ','')+'\t'+str(qs_num+1)
		temp_example+='\t'+temp_ans
		if temp_example not in examples:
			examples.append(temp_example)
			pbar.update(1)
	pbar.close()		
	return examples	
	
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
	opfileans=str(input("\n>Enter output file name: "))

num_sentences=input("\n>How many sentences per example? e.g. 2 ") 
num_sentences=int(num_sentences)
##confirm numbers to be generated
len_names,len_locs,len_verbs,num_examples = confirm_numbers(num_sentences,len(names),len(locs),len(verbs))

names=names[:len_names]
locs=locs[:len_locs]
verbs=verbs[:len_verbs]

examples=generate(num_sentences,names,locs,verbs,num_examples)
fo=open(opfile,'w')
for e in examples:
	fo.write(e+'\n')

print('Output written to '+opfile)
fo.close()

		
		

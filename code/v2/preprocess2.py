import re
from functools import reduce
import numpy as np
import spacy
import pandas as pd
import csv
nlp = spacy.load("en_core_web_sm")

gram_base=["pos_", "tag_", "ent_type_", "is_alpha", "is_stop", "is_digit", "is_lower", "is_upper","is_punct", "is_left_punct", "is_right_punct", "is_bracket", "is_quote", "dep_", "head.pos_", "head.head.pos_"]
padbe_2=[False if gb.startswith('is') else 'None' for gb in gram_base]
padb_1=['<BEG>',-1]
pade_1=['<END>',-1]
padbe_3=[0,0]

def pad_stories(gram_f, glove_f, embeddingdim,maxlen):
	new_gram_f=[]
	padg_2=[0]*embeddingdim
	padblank_gram=padb_1+padbe_2+[0]
	padblank_gram[0]='<PAD>'
	padblank_glove=padb_1+padg_2
	padblank_glove[0]='<PAD>'
	new_glove_f=[]
	for story in gram_f:
		temp_story=story
		print(maxlen,len(story),maxlen-len(story))
		ds
		for r in range(maxlen-len(story)):
			temp_story+=padblank_gram
		new_gram_f+=[temp_story]
	for story in glove_f:
		temp_story=story
		for r in range(maxlen-len(story)):
			temp_story+=padblank_glove
		new_glove_f+=[temp_story]
	return new_gram_f, new_glove_f
		
		
def create_features_headers(headertype=1,embeddingdim=0,context_length=0):
	'''
	headertype=1 == grammatical headers
	
	headertype=2 == glove headers
	'''
	context_headers_gram=["text","word_idx"]
	
	context_headers_glove=['text','word_idx']

	for cl in range(context_length,0,-1):
		context_headers_glove+=['g_'+str(i)+'_wb'+str(cl) for i in range(0,embeddingdim)]
		context_headers_gram+=[i+'_wb'+str(cl) for i in gram_base]
		context_headers_glove+=['query_word'+'_wb'+str(cl)]
		context_headers_gram+=['query_word'+'_wb'+str(cl)]

	context_headers_glove+=['g_'+str(i)+'_wt' for i in range(0,embeddingdim)]
	context_headers_glove+=['query_word'+'_wt']
	context_headers_gram+=[i+'_wt' for i in gram_base]
	context_headers_gram+=['query_word'+'_wt']
	

	for cl in range(1,context_length+1):
		context_headers_glove+=['g_'+str(i)+'_wa'+str(cl) for i in range(0,embeddingdim)]
		context_headers_gram+=[i+'_wa'+str(cl) for i in gram_base]
		context_headers_glove+=['query_word'+'_wa'+str(cl)]
		context_headers_gram+=['query_word'+'_wa'+str(cl)]
	
	context_headers_glove+=['label']
	context_headers_gram+=['label']
	
	if headertype==1:
		return context_headers_gram
	if headertype==2:
		return context_headers_glove
		
def sent_to_gram_features(sent):
	doc = nlp(sent)
	sentfeatures=[]
	for token in doc:
		tokfeatures=[token.text, token.idx, token.pos_, token.tag_, token.ent_type_, token.is_alpha, token.is_stop, token.is_digit, token.is_lower, token.is_upper,token.is_punct, token.is_left_punct, token.is_right_punct, token.is_bracket, token.is_quote, token.dep_, token.head.pos_, token.head.head.pos_]
		sentfeatures.append(tokfeatures)
	return(sentfeatures)	

def sent_to_glove_features(sent, glove_embeddings, embeddingdim):
	doc = nlp(sent)
	sentfeatures=[]
	for token in doc:
		tempglove=[token.text, token.idx]
		try:
			tempglove+= glove_embeddings.loc[token.text.lower()].values.tolist()
		except Exception as e:
			tempglove+=[0]*embeddingdim
		sentfeatures.append(tempglove)
	return(sentfeatures)	

def story_to_gram_features_sentwise(story):
	story_features=[]
	query=story[1]
	answer=story[2]
	story=story[0]
	sents=story.split('<END><BEG>')
	for sentence in sents:
		sentence=sentence.replace('<BEG>','').replace('<END>','')
		#print('Sentence',sentence)
		sent_features=sent_to_gram_features(sentence)
		padb=padb_1+padbe_2
		pade=pade_1+padbe_2		
		sent_features=[padb]+sent_features+[pade]
		'''for r in range(maxlen-len(sent_features)):
			sent_features+=padblank'''
		sent_features=[s+[1] if s[0] in query else s+[0] for s in sent_features]  ##query
		story_features+=sent_features
	return story_features, answer, len(story_features)
	
def story_to_glove_features_sentwise(story, glove_embeddings, embeddingdim):
	padg_2=[0]*embeddingdim
	story_features=[]
	query=story[1]
	answer=story[2]
	story=story[0]
	sents=story.split('<END><BEG>')
	for sentence in sents:
		sentence=sentence.replace('<BEG>','').replace('<END>','')
		#print('Sentence',sentence)
		sent_features=sent_to_glove_features(sentence,glove_embeddings, embeddingdim)
		#print('Orig',len(sent_features),len(sent_features[0]))
		padb=padb_1+padg_2
		pade=pade_1+padg_2
		sent_features=[padb]+sent_features+[pade]
		'''for r in range(maxlen-len(sent_features)):
			sent_features+=padblank'''
		sent_features=[s+[1] if s[0] in query else s+[0] for s in sent_features]  ##query
		story_features+=sent_features
	return story_features, answer, len(story_features)
	

def story_to_gram_features(story,context_length):
	story_features=[]
	query=story[1]
	answer=story[2]
	story=story[0]
	sents=story.split('<END><BEG>')
	#print('Sents', sents)
	for sentence in sents:
		sentence=sentence.replace('<BEG>','').replace('<END>','')
		#print('Sentence',sentence)
		sent_features=sent_to_gram_features(sentence)
		#print('Orig',len(sent_features),len(sent_features[0]))
		padb=padb_1+padbe_2
		pade=pade_1+padbe_2
		sent_features=[padb]+sent_features+[pade]
		#print('Pad',len(sent_features),len(sent_features[0]))
		sent_features=[s+[1] if s[0] in query else s+[0] for s in sent_features]  ##query
		#print('Q ',len(sent_features),len(sent_features[0]) )

		sent_features=[s+[1] if s[0] in answer else s+[0] for s in sent_features] ##answer
		#print('A ',len(sent_features),len(sent_features[0]) )
		story_features+=sent_features
	#print('Story ',len(story_features),len(story_features[0]) )
	padded_story_features=[]
	for cl in range(context_length-1):
		padded_story_features+=[padb+padbe_3]
	padded_story_features+=story_features
	for cl in range(context_length-1):
		padded_story_features+=[pade+padbe_3]
	#print('Story ',len(padded_story_features),len(padded_story_features[0]) )
	context_story_features=[]
	for word_idx in range(context_length,context_length+len(story_features)-2):
		tempwf=padded_story_features[word_idx][:2]
		#print('target: ',word_idx,padded_story_features[word_idx][0],len(tempwf))
		for prev_word_idx in range(word_idx-context_length,word_idx):
			#print('prev: ',prev_word_idx,padded_story_features[prev_word_idx][0], len(tempwf), len(padded_story_features[prev_word_idx][2:-1]))
			tempwf+=padded_story_features[prev_word_idx][2:-1]
		#print('word: ',word_idx,padded_story_features[word_idx][0],len(tempwf))
		tempwf+=padded_story_features[word_idx][2:-1]
		for next_word_idx in range(word_idx+1,word_idx+context_length+1):
			#print('next: ',next_word_idx,padded_story_features[next_word_idx][0],len(tempwf))
			tempwf+=padded_story_features[next_word_idx][2:-1]
		#print()
		tempwf+=[padded_story_features[word_idx][-1]]
		if (tempwf[0] not in ['<BEG>', '<END>']):
			context_story_features.append(tempwf)
	#print('Story_context ',len(context_story_features),len(context_story_features[0]) )
	return context_story_features

def story_to_glove_features(story, glove_embeddings, embeddingdim,context_length):
	padg_2=[0]*embeddingdim
	story_features=[]
	query=story[1]
	answer=story[2]
	story=story[0]
	sents=story.split('<END><BEG>')
	#print('Sents', sents)
	for sentence in sents:
		sentence=sentence.replace('<BEG>','').replace('<END>','')
		#print('Sentence',sentence)
		sent_features=sent_to_glove_features(sentence,glove_embeddings, embeddingdim)
		#print('Orig',len(sent_features),len(sent_features[0]))
		padb=padb_1+padg_2
		pade=pade_1+padg_2
		sent_features=[padb]+sent_features+[pade]
		#print('Pad',len(sent_features),len(sent_features[0]))
		sent_features=[s+[1] if s[0] in query else s+[0] for s in sent_features]  ##query
		#print('Q ',len(sent_features),len(sent_features[0]) )

		sent_features=[s+[1] if s[0] in answer else s+[0] for s in sent_features] ##answer
		#print('A ',len(sent_features),len(sent_features[0]) )
		story_features+=sent_features
	#print('Story ',len(story_features),len(story_features[0]) )
	padded_story_features=[]
	for cl in range(context_length-1):
		padded_story_features+=[padb+padbe_3]
	padded_story_features+=story_features
	for cl in range(context_length-1):
		padded_story_features+=[pade+padbe_3]
	#print('Story ',len(padded_story_features),len(padded_story_features[0]) )
	context_story_features=[]
	for word_idx in range(context_length,context_length+len(story_features)-2):
		tempwf=padded_story_features[word_idx][:2]
		#print('target: ',word_idx,padded_story_features[word_idx][0],len(tempwf))
		for prev_word_idx in range(word_idx-context_length,word_idx):
			#print('prev: ',prev_word_idx,padded_story_features[prev_word_idx][0], len(tempwf), len(padded_story_features[prev_word_idx][2:-1]))
			tempwf+=padded_story_features[prev_word_idx][2:-1]
		#print('word: ',word_idx,padded_story_features[word_idx][0],len(tempwf))
		tempwf+=padded_story_features[word_idx][2:-1]
		for next_word_idx in range(word_idx+1,word_idx+context_length+1):
			#print('next: ',next_word_idx,padded_story_features[next_word_idx][0],len(tempwf))
			tempwf+=padded_story_features[next_word_idx][2:-1]
		#print()
		tempwf+=[padded_story_features[word_idx][-1]]
		if (tempwf[0] not in ['<BEG>', '<END>']):
			context_story_features.append(tempwf)
	#print('Story_context ',len(context_story_features),len(context_story_features[0]) )
	return context_story_features


def parse_stories(lines):
    '''
    - Parse stories provided in the bAbI tasks format
    - A story starts from line 1 to line 15. Every 3rd line,
      there is a question &amp;amp;amp;amp;amp; answer.
    - Function extracts sub-stories within a story and
      creates tuples
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip() #line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            # reset story when line ID=1 (start of new story)
            story = []
        if '\t' in line:
            # this line is tab separated Q, A &amp;amp;amp;amp;amp; support fact ID
            q, a, supporting = line.split('\t')
            # tokenize the words of question
            #q = tokenize(q)
            # Provide all the sub-stories till this question
            substory = [x for x in story if x]
            # A story ends and is appended to global story data-set
            data.append((substory, q, a))
            story.append('')
        else:
            # this line is a sentence of story
            sent='<BEG>'+line+'<END>' #sent = tokenize(line)
            story.append(sent)
    return data
 
def get_stories(f):
    '''
    argument: filename
    returns list of all stories in the argument data-set file
    '''
    f=open(f,'r')
    # read the data file and parse 10k stories
    data = parse_stories(f.readlines())
    # lambda func to flatten the list of sentences into one list
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    # creating list of tuples for each story
    data = [(flatten(story), q, answer) for story, q, answer in data]
    return data

def load_meanbinarized_glove(EMBEDDING_DIM):
	glove_file="../../data/glove/glove.6B."+str(EMBEDDING_DIM)+"d.txt"

	print('loading glove, takes time...')
	words = pd.read_csv(glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
	mean=words.mean(axis = 0)
	for c in words.columns:
		words[c] = (words[c] <= mean[c]).astype(int)
	print('GloVe loaded')
	return words

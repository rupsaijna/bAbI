import re
from functools import reduce
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")

gram_base=["pos_", "tag_", "ent_type_", "is_alpha", "is_stop", "is_digit", "is_lower", "is_upper","is_punct", "is_left_punct", "is_right_punct", "is_bracket", "is_quote", "dep_", "head.pos_", "head.head.pos_"]
padbe_2=[False if gb.startswith('is') else 'None' for gb in gram_base]
padb_1=['<BEG>',-1]
pade_1=['<END>',-1]

def create_features_headers(EMBEDDING_DIM,context_length):
	context_headers_gram=["text","word_idx"]
	
	context_headers_glove=['text','word_idx']

	for cl in range(context_length,0,-1):
		context_headers_glove+=['g_'+str(i)+'_wb'+str(cl) for i in range(0,EMBEDDING_DIM)]
		context_headers_gram+=[i+'_wb'+str(cl) for i in gram_base]

	context_headers_glove+=['g_'+str(i)+'_w' for i in range(0,EMBEDDING_DIM)]
	context_headers_gram+=[i+'_w' for i in gram_base]

	for cl in range(1,context_length+1):
		context_headers_glove+=['g_'+str(i)+'_wa'+str(cl) for i in range(0,EMBEDDING_DIM)]
		context_headers_gram+=[i+'_wa'+str(cl) for i in gram_base]
		
def sent_to_gram_features(sent):
	doc = nlp(sent)
	sentfeatures=[]
	for token in doc:
		tokfeatures=[token.text, token.idx, token.pos_, token.tag_, token.ent_type_, token.is_alpha, token.is_stop, token.is_digit, token.is_lower, token.is_upper,token.is_punct, token.is_left_punct, token.is_right_punct, token.is_bracket, token.is_quote, token.dep_, token.head.pos_, token.head.head.pos_]
		sentfeatures.append(tokfeatures)
	return(sentfeatures)	

def story_to_features(story):
	story_features=[]
	query=story[1]
	answer=story[2]
	story=story[0]
	sents=story.split('<END><BEG>')
	print('Sents', sents)
	for sentence in sents:
		sentence=sentence.replace('<BEG>','').replace('<END>','')
		print('Sentence',sentence)
		sent_features=sent_to_gram_features(sentence)
		print(sent_features)
		padb=padb_1+padbe_2
		pade=pade_1+padbe_2
		sent_features=[padb]+sent_features+[pade]
		print(sent_features)
		sent_features=[s.append(1) if s[0] in query else s.append(0) for s in sent_features]  ##query
		sent_features=[s.append(1) if s[0] in answer else s.append(0) for s in sent_features]  ##answer
		
		story_features.append(sent_features)
	return story_features

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
	glove_file="../data/glove/glove.6B."+str(EMBEDDING_DIM)+"d.txt"

	print('loading glove, takes time...')
	words = pd.read_csv(glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
	mean=words.mean(axis = 0)
	for c in words.columns:
		words[c] = (words[c] <= mean[c]).astype(int)
	print('GloVe loaded')
	return words

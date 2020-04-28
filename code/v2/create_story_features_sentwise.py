from preprocess2 import *
import pandas as pd
import pickle

EMBEDDING_DIM = 100

challenge = '../../data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
print('Extracting stories for the challenge: single_supporting_fact')


glove_embeddings=load_meanbinarized_glove(EMBEDDING_DIM) #load_binarized_glove()

#gram_headers=create_features_headers(headertype=1,context_length=CONTEXT_LENGTH)
#glove_headers=create_features_headers(headertype=2,embeddingdim=EMBEDDING_DIM,context_length=CONTEXT_LENGTH)

ftype='train'
# Extracting train stories
train_stories = get_stories(challenge.format(ftype))

print('Number of training stories:', len(train_stories))

train_features_gram=[]
train_features_glove=[]
train_labels=[]
MAXLEN=0
counter=1
for trs in train_stories:
	print('Train Story:',counter)
	temp_features, temp_answer, temp_len=story_to_gram_features_sentwise(trs)
	train_features_gram+=[temp_features]
	train_labels.append(temp_answer)
	if temp_len>MAXLEN:
		MAXLEN=temp_len
	temp_features, temp_answer, temp_len=story_to_glove_features_sentwise(trs,glove_embeddings, EMBEDDING_DIM)
	train_features_glove+=[temp_features]
	counter+=1
	

################################################################

# Extracting test stories
ftype='test'
test_stories = get_stories(challenge.format(ftype))
print('Number of test stories:', len(test_stories))
test_features_gram=[]
test_features_glove=[]
test_labels=[]
counter=1
for trs in test_stories:
	print('Test Story:',counter)
	temp_features, temp_answer, temp_len=story_to_gram_features_sentwise(trs)
	test_features_gram+=[temp_features]
	test_labels.append(temp_answer)
	if temp_len>MAXLEN:
		MAXLEN=temp_len
	temp_features, temp_answer, temp_len=story_to_glove_features_sentwise(trs,glove_embeddings, EMBEDDING_DIM)
	test_features_glove+=[temp_features]
	counter+=1
	
print("MAXLEN", MAXLEN)
print(len(test_features_gram), len(test_features_gram[0]), len(test_features_gram[0][0]))
print(len(test_features_glove), len(test_features_glove[0]), len(test_features_glove[0][0]))
test_features_gram, test_features_glove= pad_stories(test_features_gram, test_features_glove, EMBEDDING_DIM, MAXLEN)
print('padded')
print(len(test_features_gram), len(test_features_gram[0]), len(test_features_gram[0][0]))
print(len(test_features_glove), len(test_features_glove[0]), len(test_features_glove[0][0]))

savename='../../pickles/spacy/nonbinarized_features_sentence'+'_'+ftype
with open(savename+'_gram.pkl', 'wb') as f:
	pickle.dump(test_features_gram, f)
print('Saved: ',savename+'_gram.pkl')

with open(savename+'_glove.pkl', 'wb') as f:
	pickle.dump(test_features_glove, f)
print('Saved: ',savename+'_glove.pkl')

with open(savename+'_labels.pkl', 'wb') as f:
	pickle.dump(test_labels, f)
print('Saved: ',savename+'_labels.pkl')


ftype='train'
print(len(train_features_gram), len(train_features_gram[0]), len(train_features_gram[0][0]))
print(len(train_features_glove), len(train_features_glove[0]), len(train_features_glove[0][0]))
train_features_gram, train_features_glove= pad_stories(train_features_gram, train_features_glove, EMBEDDING_DIM, MAXLEN)
print('padded')
print(len(train_features_gram), len(train_features_gram[0]), len(train_features_gram[0][0]))
print(len(train_features_glove), len(train_features_glove[0]), len(train_features_glove[0][0]))


savename='../../pickles/spacy/nonbinarized_features_sentence'+'_'+ftype
with open(savename+'_gram.pkl', 'wb') as f:
	pickle.dump(train_features_gram, f)
print('Saved: ',savename+'_gram.pkl')

with open(savename+'_glove.pkl', 'wb') as f:
	pickle.dump(train_features_glove, f)
print('Saved: ',savename+'_glove.pkl')

with open(savename+'_labels.pkl', 'wb') as f:
	pickle.dump(train_labels, f)
print('Saved: ',savename+'_labels.pkl')

from preprocess2 import *
import pandas as pd

EMBEDDING_DIM = 100
CONTEXT_LENGTH=5

challenge = '../../data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
print('Extracting stories for the challenge: single_supporting_fact')


glove_embeddings=load_meanbinarized_glove(EMBEDDING_DIM) #load_binarized_glove()

gram_headers=create_features_headers(headertype=1,context_length=CONTEXT_LENGTH)
glove_headers=create_features_headers(headertype=2,embeddingdim=EMBEDDING_DIM,context_length=CONTEXT_LENGTH)

ftype='train'
# Extracting train stories
train_stories = get_stories(challenge.format(ftype))
print('Number of training stories:', len(train_stories))
train_features_gram=[]
train_features_glove=[]
counter=1
for trs in train_stories:
	print('Train Story:',counter)
	train_features_gram+=story_to_gram_features(trs,context_length=CONTEXT_LENGTH)
	train_features_glove+=story_to_glove_features(trs,glove_embeddings, EMBEDDING_DIM,context_length=CONTEXT_LENGTH)
	counter+=1
	
train_features_gram=pd.DataFrame(train_features_gram, columns=gram_headers)
train_features_glove=pd.DataFrame(train_features_glove, columns=glove_headers)
print(train_features_gram.shape)
print(train_features_glove.shape)

savename='../../pickles/spacy/nonbinarized_features_context'+str(CONTEXT_LENGTH)+'_'+ftype
train_features_gram.to_pickle(savename+'_gram.pkl')
print('Saved: ',savename+'_gram.pkl')

train_features_glove.to_pickle(savename+'_glove.pkl')
print('Saved: ',savename+'_glove.pkl')

################################################################

# Extracting test stories
ftype='test'
test_stories = get_stories(challenge.format(ftype))
print('Number of test stories:', len(test_stories))
test_features_gram=[]
test_features_glove=[]
counter=1
for trs in test_stories:
	print('Test Story:',counter)
	test_features_gram+=story_to_gram_features(trs,context_length=CONTEXT_LENGTH)
	test_features_glove+=story_to_glove_features(trs,glove_embeddings, EMBEDDING_DIM,context_length=CONTEXT_LENGTH)
	counter+=1
  
test_features_gram=pd.DataFrame(test_features_gram, columns=gram_headers)
print(test_features_gram.shape)
test_features_glove=pd.DataFrame(test_features_glove, columns=glove_headers)
print(test_features_glove.shape)

savename='../../pickles/spacy/nonbinarized_features_context'+str(CONTEXT_LENGTH)+'_'+ftype
test_features_gram.to_pickle(savename+'_gram.pkl')
print('Saved: ',savename+'_gram.pkl')

test_features_glove.to_pickle(savename+'_glove.pkl')
print('Saved: ',savename+'_glove.pkl')

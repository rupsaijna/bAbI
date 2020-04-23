from preprocess import *
import pandas as pd

EMBEDDING_DIM = 100
context_length=2

challenge = '../../data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
print('Extracting stories for the challenge: single_supporting_fact')

gram_headers=create_features_headers(headertype=1)

ftype='train'
# Extracting train stories
train_stories = get_stories(challenge.format(ftype))
print('Number of training stories:', len(train_stories))
train_features_gram=[]
counter=1
for trs in train_stories:
	print('Story:',counter,' ',trs)
	train_features_gram+=story_to_gram_features(trs)
	counter+=1
	
print(len(train_features_gram), len(train_features_gram[0]), len(gram_headers))
train_features_gram=pd.DataFrame(train_features_gram, columns=gram_headers)
savename='../../pickles/spacy/nonbinarized_features_'+ftype
train_features_gram.to_pickle(savename+'_gram.pkl')
print('Saved: ',savename+'_gram.pkl')

################################################################

# Extracting test stories
ftype='test'
test_stories = get_stories(challenge.format(ftype))
print('Number of test stories:', len(test_stories))
test_features_gram=[]
counter=1
for trs in test_stories:
	print('Story:',counter,' ',trs)
	test_features_gram+=story_to_gram_features(trs)
	counter+=1
  
test_features_gram=pd.DataFrame(test_features_gram, columns=gram_headers)
savename='../../pickles/spacy/nonbinarized_features_'+ftype
test_features_gram.to_pickle(savename+'_gram.pkl')
print('Saved: ',savename+'_gram.pkl')

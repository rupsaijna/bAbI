from preprocess2 import *

EMBEDDING_DIM = 100
challenge = '../../data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
print('Extracting stories for the challenge: single_supporting_fact')

#glove_embeddings=load_meanbinarized_glove(EMBEDDING_DIM) #load_binarized_glove()


# Extracting train stories
train_stories = get_stories(challenge.format('train'))
# Extracting test stories
test_stories = get_stories(challenge.format('test'))

print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))

train_features=[]
for trs in train_stories:
	print('Story:', trs)
	#temp=story_to_glove_features(trs, glove_embeddings, EMBEDDING_DIM)
	temp=story_to_gram_features(trs,context_length=2)
	train_features+=temp
	break

print()

for ff in train_features:
	print(ff)
	
print(len(train_features), len(train_features[0]))


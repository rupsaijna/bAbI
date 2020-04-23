from preprocess import *

challenge = '../../data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
print('Extracting stories for the challenge: single_supporting_fact')


# Extracting train stories
train_stories = get_stories(challenge.format('train'))
# Extracting test stories
test_stories = get_stories(challenge.format('test'))

print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('Story 1:', train_stories[0])
print('Story 2:', train_stories[1])
print('Story 3:', train_stories[2])


challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
print('Extracting stories for the challenge: single_supporting_fact_10k')
# Extracting train stories
train_stories = get_stories(tar.extractfile(challenge.format('train')))
# Extracting test stories
test_stories = get_stories(tar.extractfile(challenge.format('test')))

print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print(train_stories[0])

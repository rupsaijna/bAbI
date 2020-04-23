from preproc import *

challenge = '../data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
print('Extracting stories for the challenge: single_supporting_fact')

# Extracting train stories
train_stories = get_stories(challenge.format('train'))
# Extracting test stories
test_stories = get_stories(challenge.format('test'))

print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('Story 1:', train_stories[0])
print('Story 2:', train_stories[1])
dsf

# creating vocabulary of words in train and test set
vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
 
# sorting the vocabulary
vocab = sorted(vocab)
 
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
 
# calculate maximum length of story
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
 
# calculate maximum length of question/query
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
 
# creating word to index dictionary
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
 
# creating index to word dictionary
idx_word = dict((i+1, c) for i,c in enumerate(vocab))
 
# vectorize train story, query and answer sentences/word using vocab
inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               word_idx,
                                                               story_maxlen,
                                                               query_maxlen)
# vectorize test story, query and answer sentences/word using vocab
inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            word_idx,
                                                            story_maxlen,
                                                            query_maxlen)

print('-------------------------')
print('Vocabulary:\n',vocab,"\n")
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-------------------------')

print('-------------------------')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('input train sample', inputs_train[0,:])
print('-------------------------')

print('-------------------------')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('query train sample', queries_train[0,:])
print('-------------------------')

print('-------------------------')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('answer train sample', answers_train[0,:])
print('-------------------------')

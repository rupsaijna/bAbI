from preproc import *
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

challenge = '../data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
print('Extracting stories for the challenge: single_supporting_fact')

# Extracting stories
train_stories = get_stories(challenge.format('train'))
test_stories = get_stories(challenge.format('test'))

print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))

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

bitlength=story_maxlen.bit_length()
new_inputs_train=[]
for inp in inputs_train:
  ip=[]
  for w in inp:
    t=list(bin(int(w))[2:].zfill(bitlength))
    ip+=t
  ip=[int(t) for t in ip]
  new_inputs_train.append(ip)
new_inputs_train= np.asarray(new_inputs_train)

new_inputs_test=[]
for inp in inputs_test:
  ip=[]
  for w in inp:
    t=list(bin(int(w))[2:].zfill(bitlength))
    ip+=t
  ip=[int(t) for t in ip]
  new_inputs_test.append(ip)
  
new_inputs_test= np.asarray(new_inputs_test)

bitlength=query_maxlen.bit_length()
new_queries_train=[]
for inp in queries_train:
  ip=[]
  for w in inp:
    t=list(bin(int(w))[2:].zfill(bitlength))
    ip+=t
  ip=[int(t) for t in ip]
  new_queries_train.append(ip)

new_queries_train= np.asarray(new_queries_train)

new_queries_test=[]
for inp in queries_test:
  ip=[]
  for w in inp:
    t=list(bin(int(w))[2:].zfill(bitlength))
    ip+=t
  ip=[int(t) for t in ip]
  new_queries_test.append(ip)

new_queries_test= np.asarray(new_queries_test)

match_train = []
for q in new_queries_train:
    for s in new_inputs_train:
        m=np.dot([input_encoded_m, question_encoded], axes=-1, normalize=False)
        match_train.append(m)
match_train=np.asarray(match_train)
print(match_train.shape)

'''
tm = MultiClassTsetlinMachine(800, 40, 5.0)
tm.fit(new_inputs_train, new_queries_train, epochs=1)
tm_results=tm.predict(X_test) == Y_test).mean()

print(tm_results)
'''

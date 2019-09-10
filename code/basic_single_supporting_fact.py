from keras.layers.embeddings import Embedding
#from keras.layers.core import Dense, Merge
from keras.layers import recurrent
#from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from functools import reduce
import numpy as np
import re
np.random.seed(1337)

challenge = '../data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
print('\nExtracting stories for the challenge: single_supporting_fact')

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    
def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data
    
def get_stories(fn, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    f=open(fn)
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(vocab_size)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

def binarize(listy, bitlength):
    new_listy=[]
    for ls in listy:
        ns=[]
        for w in ls:
            t=list(bin(int(w))[2:].zfill(bitlength))
            ns+=t
        ns=[int(t) for t in ns]
        new_listy.append(ns)
    return np.asarray(new_listy)
           
train = get_stories(challenge.format('train'))
test = get_stories(challenge.format('test'))

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

X, Xq, Y = vectorize_stories(train)
tX, tXq, tY = vectorize_stories(test)

bitlength=vocab_size.bit_length()
X=binarize(X,bitlength)
Xq=binarize(Xq,bitlength)
tX=binarize(tX,bitlength)
tXq=binarize(tXq,bitlength)
Y=[int(y) for y in Y]
tY=[int(y) for y in tY]

print('\n\nvocab = {}'.format(vocab))
print('X.shape = {}'.format(X.shape))
print('Xq.shape = {}'.format(Xq.shape))
print('Y.shape = {}'.format(Y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))


print(train[0], X[0], Xq[0],Y[0])

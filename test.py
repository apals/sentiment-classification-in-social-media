from __future__ import absolute_import
from textblob.classifiers import NaiveBayesClassifier
from itertools import chain

import nltk

from textblob.compat import basestring
from textblob.decorators import cached_property
from textblob.exceptions import FormatError
from textblob.tokenizers import word_tokenize
from textblob.utils import strip_punc, is_filelike
import textblob.formats as formats
train = [('This is an amazing place!', 'pos'),
        ('I feel very good about these beers.', 'pos'),
        ('This is my best work.', 'pos'),
        ("What an awesome view", 'pos'),
        ('I do not like this restaurant', 'neg'),
        ('I am tired of this stuff.', 'neg'),
        ("I can't deal with this", 'neg'),
        ('He is my sworn enemy!', 'neg'),
        ('My boss is horrible.', 'neg') ] 
test = [
        ('The beer was good.', 'pos'),
        ('I do not enjoy my job', 'neg'),
        ("I ain't feeling dandy today.", 'neg'),
        ("I feel amazing!", 'pos'),
        ('Gary is a friend of mine.', 'pos'),
        ("I can't believe I'm doing this.", 'neg') ] 
f = open("twitter/tweets_GroundTruth-parsed.txt")
i = 0
s1 = ""
s2 = ""
tup = (s1, s2)
for line in f:
    if i % 2 == 0:
        s1 = line
    else:
        s2 = line
        tup = (s1, s2)
        train.append(tup)


def _get_words_from_dataset(dataset):
    """Return a set of all words in a dataset.
   :param dataset: A list of tuples of the form ``(words, label)`` where
   ``words`` is either a string of a list of tokens.
   """
# Words may be either a string or a list of tokens. Return an iterator
# of tokens accordingly
def tokenize(words):
    if isinstance(words, basestring):
        return word_tokenize(words, include_punc=False)
    else:
        return words
    all_words = chain.from_iterable(tokenize(words) for words, _ in dataset)
    return set(all_words)

def _get_document_tokens(document):
    if isinstance(document, basestring):
        tokens = set((strip_punc(w, all=False) for w in word_tokenize(document, include_punc=False)))
    else:
        tokens = set(strip_punc(w, all=False) for w in document)
        return tokens[docs]

    def basic_extractor(document, train_set):
        word_features = _get_words_from_dataset(train_set)
        tokens = _get_document_tokens(document)
        features = dict(((u'contains({0})'.format(word), (word in tokens)) for word in word_features))
        return features

def contains_extractor(document):
    tokens = _get_document_tokens(document)
    features = dict((u'contains({0})'.format(w), True) for w in tokens)
    return features

print contains_extractor(train)

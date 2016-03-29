import pickle
from nltk.probability import DictionaryProbDist
from nltk import NaiveBayesClassifier
from nltk import FreqDist, ConditionalFreqDist
from nltk import BigramAssocMeasures
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

train = []
f = open("twitter/tweets_GroundTruth-parsed.txt")
print "creating data set"
i = 0
s1 = ""
s2 = ""
tup = (s1, s2)
for line in f:
    if i % 2 == 0:
        s1 = line.split()
    else:
        s2 = line
        tup = (s1, s2)
        train.append(tup)
    i += 1

train_docs = train[:3359]
test_docs = train[3359:]

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in train_docs])
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=6)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(train_docs)
testing_set = sentim_analyzer.apply_features(test_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
for key, value in sorted(sentim_analyzer.evaluate(testing_set).items()):
    print('{0}: {1}'.format(key, value))
#f = open('naive_bayes_trained_with_80_percent.pickle', 'wb')
#pickle.dump(classifier, f)
#f.close()'''

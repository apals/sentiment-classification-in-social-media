from nltk.probability import DictionaryProbDist
from nltk import FreqDist, ConditionalFreqDist
from nltk import BigramAssocMeasures
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import pickle
from nltk.classify.svm import SvmClassifier
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC


train = []


#f = open("twitter/lol.txt", "r")
f = open("twitter/tweets_GroundTruth-parsed.txt")
print "creating data set"
i = 0
s1 = ""
s2 = ""
tup = (s1, s2)
for line in f:
    if i > 6718:
        break
    if i % 2 == 0:
        s1 = line.split()
    else:
        s2 = line
        tup = (s1, s2)
        train.append(tup)
    i += 1

print train
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in train])
print all_words_neg
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg)
print unigram_feats
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
svmc = SklearnClassifier(SVC(), sparse=False).train(train)
classifier = sentim_analyzer.train(trainer, training_set)

f = open('svm_trained_with_80_percent.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

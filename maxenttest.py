import pickle
from nltk.probability import DictionaryProbDist
from nltk import MaxentClassifier
from nltk import FreqDist, ConditionalFreqDist
from nltk import BigramAssocMeasures
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import pickle


#f = open("maxent_trained_with_80_percent.pickle", "rb")
#meclassifier = pickle.load(f)
#f.close()

train = []


#f = open("twitter/lol.txt", "r")
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


min_freqs = [1, 2, 3, 4, 5, 6]

for min_freqq in min_freqs:

    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in train_docs])
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, top_n=10, min_freq=min_freqq)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    training_set = sentim_analyzer.apply_features(train_docs)
    testing_set = sentim_analyzer.apply_features(test_docs)

    trainer = MaxentClassifier.train
    classifierme = sentim_analyzer.train(trainer, training_set)

    f = open('results/maxent/maxent_top_n_200_min_freq_' + str(min_freqq) + '.txt', 'w')
    for key, value in sorted(sentim_analyzer.evaluate(testing_set, classifier=classifierme).items()):
        print('{0}: {1}'.format(key, value))
        f.write('{0}: {1}'.format(key, value))
    f.close()

#f = open('maxent_trained_with_80_percent_2.pickle', 'wb')
#pickle.dump(classifierme, f)
#f.close()



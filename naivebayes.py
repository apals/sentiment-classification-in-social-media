from nltk.probability import DictionaryProbDist
from nltk import NaiveBayesClassifier as nbc
from nltk import FreqDist, ConditionalFreqDist
from nltk import BigramAssocMeasures
from nltk.tokenize import word_tokenize
from itertools import chain

#train = [('I love this sandwich.', 'pos'),
 #       ('This is an amazing place!', 'pos'),
 #       ('I feel very good about these beers.', 'pos'),
 #       ('This is my best work.', 'pos'),
 #       ("What an awesome view", 'pos'),
 #       ('I do not like this restaurant', 'neg'),
#        ('I am tired of this stuff.', 'neg'),
#        ("I can't deal with this", 'neg'),
#        ('He is my sworn enemy!', 'neg'),
 #       ('My boss is horrible.', 'neg')]

train = []
test_sentence = "I love this bar"


#f = open("twitter/lol.txt", "r")
f = open("twitter/tweets_GroundTruth-parsed.txt")
print "creating data set"
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
    i += 1
print "done with data set"


vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in train]))
print "done with data set2"



#feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in train]

f2 = []
i = 0
for sentence, tag in train:
    print i
    print len(train)
    m = ({i: (i in (word_tokenize(sentence))) for i in vocabulary}, tag)
    f2.append(m)


print feature_set
print "-----------------------------------------------------"
print f2
print "done with data set3"
classifier = nbc.train(feature_set)

test_sentence = "This is the best band I've ever heard!"
featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary}

print "test_sent:",test_sentence
print "tag:",classifier.classify(featurized_test_sentence)

from nltk.probability import DictionaryProbDist
from nltk import NaiveBayesClassifier
from nltk import FreqDist, ConditionalFreqDist
from nltk import BigramAssocMeasures
from nltk.tokenize import word_tokenize

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

all_words = set()
i = 0
for passage in train:
    i += 1
#    print i
#    print passage
    for word in word_tokenize(passage[0]):
        all_words.add(word.lower())

#all_words = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
print "done with all_words"
#t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]

t = []
i = 0
for word in all_words:
    i += 1
    print i
    print len(all_words)
    for x in train:
        m = ({word: (word in word_tokenize(x[0]))}, x[1])

print "about to train"
classifier = NaiveBayesClassifier.train(t)

test_sent_features = {word.lower(): (word in word_tokenize(test_sentence.lower())) for word in all_words}
print classifier.classify(test_sent_features)

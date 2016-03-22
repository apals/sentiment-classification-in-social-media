from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentences = ["Vader is smart, handsome and funny!"]

sid = SentimentIntensityAnalyzer()
f = open("twitter/tweets_GroundTruth-parsed.txt", "r")

pos = 0
neg = 0
neu = 0
tot = 0
i = 0

correct = 0.0
incorrect = 0.0


for line in f:
    if i % 2 == 0:
        ss = sid.polarity_scores(line)
        #for k in sorted(ss):
            #print('{0}: {1}, '.format(k, ss[k]))
        if ss["compound"] > 0.4:
            #print line
            #print ss["compound"]
            pos += 1
            polarity = "\"positive\""
        elif ss["compound"] < -0.2:
            neg += 1
            polarity = "\"negative\""
        else:
            neu += 1
            polarity = "\"neutral\""

        oldline = line
    else:

        if line.strip() == polarity:
            correct += 1 
        else:
            incorrect += 1
            print "should be: " + line.strip()
            print "was: " + polarity
            print oldline
    i += 1
print pos
print neg
print neu
print "-------------"
print correct
print incorrect
print 100*(correct)/(correct+incorrect)

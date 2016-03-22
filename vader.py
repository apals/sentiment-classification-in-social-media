from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentences = ["Vader is smart, handsome and funny!"]

sid = SentimentIntensityAnalyzer()
f = open("twitter/twitter-corpus-parsed.txt", "r")

pos = 0
neg = 0
neu = 0
tot = 0
for line in f:
    ss = sid.polarity_scores(line)
    if len(line) <= 1:
        print line
    #print('{0}: {1}'.format("compound", ss["compound"]))
    if ss["compound"] > 0.585:
        pos += 1
    elif ss["compound"] <= -0.1531:
        neg += 1
    else:
        neu += 1

    tot += 1

print pos
print neg
print neu


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentences = ["Vader is smart, handsome and funny!"]

sid = SentimentIntensityAnalyzer()
f = open("twitter/twitter-corpus-parsed.txt", "r")

pos = 0
neg = 0
neu = 0
tot = 0
i = 0
polarity = "hej"

correct = 0
incorrect = 0

oldline = "hejsan"

print polarity
for line in f:
    if i % 2 == 0:
        ss = sid.polarity_scores(line)
        if ss["compound"] > 0.585:
            pos += 1
            polarity = "positive"
        elif ss["compound"] <= -0.1531:
            neg += 1
            polarity = "negative"
        else:
            neu += 1
            polarity = "neutral"

        oldline = line
    else:

        if line.strip() == polarity:
            correct += 1 
        else:
            incorrect += 1
            print oldline
    i += 1
print neg
print neu
print pos
print "-------------"
print correct
print incorrect

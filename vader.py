from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentences = ["Vader is smart, handsome and funny!"]

sid = SentimentIntensityAnalyzer()
f = open("twitter/tweets_GroundTruth-parsed.txt", "r")

pos = 0.0
neg = 0.0
neu = 0.0
tot = 0.0
i = 0

correct = 0.0
incorrect = 0.0

POS = 0
NEG = 1
NEU = 2
predictedval = -1
m = [[0 for x in range(3)] for x in range(3)] 

for line in f:
    if i % 2 == 0:
        ss = sid.polarity_scores(line)
        if ss["compound"] > 0.5:
            pos += 1
            polarity = "\"positive\""
            predictedval = POS
        elif ss["compound"] < -0.3:
            neg += 1
            polarity = "\"negative\""
            predictedval = NEG
        else:
            neu += 1
            polarity = "\"neutral\""
            predictedval = NEU

        oldline = line
    else:

        if line.strip() == "\"positive\"":
            actualval = POS
        elif line.strip() == "\"negative\"":
            actualval = NEG
        elif line.strip() == "\"neutral\"":
            actualval = NEU

        m[actualval][predictedval] += 1

        if line.strip() == polarity:
            correct += 1
        else:
            incorrect += 1
    i += 1

print m[0][1]
print m
print "--------------------------------"
print "Vader found" + str(pos) + " positives"
print "There are 1997 positives"
print pos/1997
print "--------------------------------"
print "Vader found" + str(neg) + " negatives"
print "There are 917 negatives"
print neg/917
print "--------------------------------"
print "Vader found" + str(neu) + " neutrals"
print "There are 1285 neutrals"
print neu/1285
print "--------------------------------"
print "Correct: " + str(correct)
print "Incorrect: " + str(incorrect)
print "Accuracy: " + str(100*(correct)/(correct+incorrect))


TP_pos = m[0][0]
FP_pos = m[1][0] + m[2][0]
TN_pos = m[1][1] + m[1][2] + m[2][1] + m[2][2]
FN_pos = m[0][1] + m[0][2]

print TP_pos
print FP_pos
print TN_pos
print FN_pos


TP_neg = m[1][1]
FP_neg = m[0][1] + m[2][1]
TN_neg = m[0][0] + m[0][2] + m[2][0] + m[2][2]
FN_neg = m[1][0] + m[1][2]



TP_neu = m[2][2]
FP_neu = m[0][2] + m[1][2]
TN_neu = m[0][0] + m[0][1] + m[1][0] + m[1][1]
FN_neu = m[2][0] + m[2][1]



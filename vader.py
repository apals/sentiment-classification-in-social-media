from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm


#non inclusive. ranges from 0.0 - 0.5 and -0.5 - 0.0
#posrange = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#negrange = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
posrange = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
negrange = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
#posrange = [0.0, 0.1]
#negrange = [0.0, -0.1]

#negintensities = [[0 for x in range(len(posrange))] for x in range(len(negrange))]
neuintensities = [[0 for x in range(len(posrange))] for x in range(len(negrange))]
#negintensities = {}
negintensities = []
negintensities.append([])
negintensities.append([])
negintensities.append([])
negintensities.append([])
negintensities.append([])
negintensities.append([])
negintensities.append([])
negintensities.append([])
negintensities.append([])
negintensities.append([])



sid = SentimentIntensityAnalyzer()

print "-----------------------------------------------"
index = 0
for poslimit in posrange:

    for neglimit in negrange:
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
        m = [[0.0 for x in range(3)] for x in range(3)]
        f = open("twitter/tweets_GroundTruth-parsed-noemoticons.txt", "r")


        for line in f:
            if i % 2 == 0:
                ss = sid.polarity_scores(line)
                if ss["compound"] > poslimit:
                    pos += 1
                    polarity = "\"positive\""
                    predictedval = POS
                elif ss["compound"] < neglimit:
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
        '''
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
            '''

        accuracy = (correct)/(correct+incorrect)
        print "Accuracy: " + str(100*(correct)/(correct+incorrect))
        print "Positive limit: " + str(poslimit)
        print "Negative limit: " + str(neglimit)
        
        print

        TP_pos = m[0][0]
        FP_pos = m[1][0] + m[2][0]
        TN_pos = m[1][1] + m[1][2] + m[2][1] + m[2][2]
        FN_pos = m[0][1] + m[0][2]


        TP_neg = m[1][1]
        FP_neg = m[0][1] + m[2][1]
        TN_neg = m[0][0] + m[0][2] + m[2][0] + m[2][2]
        FN_neg = m[1][0] + m[1][2]



        TP_neu = m[2][2]
        FP_neu = m[0][2] + m[1][2]
        TN_neu = m[0][0] + m[0][1] + m[1][0] + m[1][1]
        FN_neu = m[2][0] + m[2][1]

        pos_recall = TP_pos/(TP_pos + FN_pos)
        pos_precision = TP_pos/(TP_pos + FP_pos)
        pos_fmeasure = 2*(pos_precision*pos_recall)/(pos_precision+pos_recall)
        print "Pos Precision: " + str(pos_precision)
        print "Pos Recall: " + str(pos_recall)
        print "Pos F-measure: " + str(pos_fmeasure)

        print

        neg_recall= TP_neg/(TP_neg + FN_neg)
        neg_precision = TP_neg/(TP_neg + FP_neg)
        neg_fmeasure = 2*(neg_precision*neg_recall)/(neg_precision+neg_recall)
        
        
        #negintensities[int(10*poslimit)][int(10*neglimit)] = neg_fmeasure
        #        negintensities[str(poslimit) + str(neglimit)] = neg_fmeasure
        
        print "neg Precision: " + str(neg_precision)
        print "neg Recall: " + str(neg_recall)
        print "neg F-measure: " + str(neg_fmeasure)

        print

        neu_recall= TP_neu/(TP_neu + FN_neu)
        neu_precision = TP_neu/(TP_neu + FP_neu)
        neu_fmeasure = 2*(neu_precision*neu_recall)/(neu_precision+neu_recall)


#        neuintensities[abs(10*poslimit), abs(10*neglimit)] = neu_fmeasure
    
        print "neu Precision: " + str(neu_precision)
        print "neu Recall: " + str(neu_recall)
        print "neu F-measure: " + str(neu_fmeasure)
        print "F-measure: " + str((neg_fmeasure + neu_fmeasure + pos_fmeasure)/3) 
        negintensities[index].append((neg_fmeasure + neu_fmeasure + pos_fmeasure)/3)
        #negintensities[index].append(accuracy)

        print "-----------------------------------------------"
        f.close()
    index += 1

'''
fig, ax = plt.subplots()
range = negintensities
print negintensities
ax.imshow(range, cmap=cm.jet, interpolation='nearest')
'''
X = [posrange, posrange, posrange, posrange, posrange, posrange, posrange, posrange, posrange, posrange]
Y = [
     [negrange[0], negrange[0], negrange[0], negrange[0], negrange[0], negrange[0]],
     [negrange[1], negrange[1], negrange[1], negrange[1], negrange[1], negrange[1]],
     [negrange[2], negrange[2], negrange[2], negrange[2], negrange[2], negrange[2]],
     [negrange[3], negrange[3], negrange[3], negrange[3], negrange[3], negrange[3]],
     [negrange[4], negrange[4], negrange[4], negrange[4], negrange[4], negrange[4]],
     [negrange[5], negrange[5], negrange[5], negrange[5], negrange[5], negrange[5]],
     [negrange[6], negrange[6], negrange[6], negrange[6], negrange[6], negrange[6]],
     [negrange[7], negrange[7], negrange[7], negrange[7], negrange[7], negrange[7]],
     [negrange[8], negrange[8], negrange[8], negrange[8], negrange[8], negrange[8]],
     [negrange[9], negrange[9], negrange[9], negrange[9], negrange[9], negrange[9]],
     ]


fig, ax = plt.subplots()

cax = ax.imshow(negintensities, extent=(np.amin(negrange), np.amax(negrange), np.amin(posrange), np.amax(posrange)),
           cmap=cm.pink, interpolation='none')
#plt.pcolormesh(X,Y,negintensities)
cbar = fig.colorbar(cax)

max = -1.0
min = 999.0
for list in negintensities:
    for m in list:
        if m > max:
            max = m
for list in negintensities:
    for m in list:
        if m < min:
            min = m

mn = min
md = (min + max) / 2
mx = max
cbar.set_ticks([mn,md,mx])
cbar.set_ticklabels(["{0:.1f}%".format(100*min),"{0:.1f}%".format(100*md),"{0:.1f}%".format(100*max)])

cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('F-measure', rotation=90)

plt.ylabel("Positive threshold")
plt.xlabel("Negative threshold")


plt.show()

'''

numrows = numcols = len(negrange)
def format_coord(x, y):
    col = int(x + 0.1)
    row = int(y + 0.1)
    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        
        #get z from your data, given x and y
        z = range[str(row) + str(col)]
        
        #this only leaves two decimal points for readability
        [x,y,z]=map("{0:.2f}".format,[x,y,z])
        
        #change return string of x,y and z to whatever you want
        return 'x='+str(x)+', y='+str(y)+', z='+str(z)+" degrees"
    else:
        return 'x=%1.4f, y=%1.4f' % (x, y)

#Set the function as the one used for display
ax.format_coord = format_coord

plt.show()
'''


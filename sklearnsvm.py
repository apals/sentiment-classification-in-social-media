from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import svm, datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.metrics import classification_report



#categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
#twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
#print twenty_train.target_names
#print len(twenty_train.data)
#print len(twenty_train.filenames)
#print("\n".join(twenty_train.data[0].split("\n")[:3]))
#print(twenty_train.target_names[twenty_train.target[0]])
#print twenty_train.target[:10]


#for t in twenty_train.target[:10]:
#    print(twenty_train.target_names[t])


target_names = ["\"positive\"", "\"negative\"", "\"neutral\""]
data = []
target = []


f = open("twitter/tweets_GroundTruth-parsed-noemoticons.txt")
i = 0

testdata = []
testtarget = []

for line in f:
    if i < 6718:
        if i % 2 == 0:
            data.append(line)
        else:
            target.append(target_names.index(line.strip()))
    else:
        if i % 2 == 0:
            testdata.append(line)
        else:
            testtarget.append(target_names.index(line.strip()))
    i += 1

#print len(data)
#print len(testdata)

#print data


count_vect = CountVectorizer(ngram_range=(1,1))
X_train_counts = count_vect.fit_transform(data)



#twentry_train.data ska vara straight up array av tweets
'''-----------------------------'''
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print X_train_tfidf.shape


#docs_new = ['God is love', 'OpenGL on the GPU is fast']
#X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_train_counts)


#twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = testdata

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', count_vect),
                     ('tfidf', tfidf_transformer),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=0.001, n_iter=3, random_state=42)),
                     ])




_ = text_clf.fit(data, target)

predicted = text_clf.predict(docs_test)


from sklearn import metrics
print metrics.accuracy_score(testtarget, predicted)
print metrics.f1_score(testtarget, predicted, average='macro')


'''
from sklearn.grid_search import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
        'clf__n_iter': (1, 2, 3, 4, 5, 6, 7)
}


gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(data[:400], target[:400])

best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

print score
'''

'''


vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(data)
test_vectors = vectorizer.transform(testdata)

# Perform classification with SVM, kernel=rbf
classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(train_vectors, target)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(test_vectors)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, target)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, target)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

# Print results in a nice table
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(testtarget, prediction_linear))
print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(testtarget, prediction_liblinear))
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# we create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

print "---------------"
print X
print "----------------"
print train_vectors
print "--------------"
print Y
# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(train_vectors, target)

# get the separating hyperplane
w = clf.coef_[0]
print "-------------------"
print "---------DiPp----------"
print w[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()
'''

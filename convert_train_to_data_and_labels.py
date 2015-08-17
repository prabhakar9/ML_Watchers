import csv
from feature_engineering import FeatureEngineering
import time
from threading import Thread
# SK-learn libraries for learning.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

def main():
    ###########################################
    # Prepare data and do feature engineering #
    ###########################################
    fe = FeatureEngineering()
    #fe.prepare_data('train.csv', 'test.csv') # Can comment this out once we've run it once succesfully
    fe.load_train_data()
    # If you're hitting memory issues, uncomment this and only call it after calling fe.load_test_data(). However, it
    # doesn't seem to help much.
    fe.load_test_data()

    # Examples of how to access the data
    print fe.train_data.shape
    print fe.train_labels.shape
    print fe.mini_train_data.shape
    print fe.mini_train_labels.shape
    print fe.dev_data.shape
    print fe.dev_labels.shape
    print fe.test_data.shape
    print fe.test_labels.shape
    print fe.submission_data.shape # Loaded by load_test_data()

    #time.sleep(15) # Used for testing memory consumption
    #print 'Testing of delete method'
    #fe.delete_train_data()
    #time.sleep(15)
    return

    ################
    # Train models #
    ################
    # can't use MNB yet - need non-negative integer data
    '''mnb = MultinomialNB(alpha = 0.01)
    mnb.fit(mini_train_data, mini_train_labels)
    mnb_predict = mnb.predict(mini_dev_data)
    mnb_f1 = metrics.f1_score(mini_dev_labels, mnb_predict)
    total_right_mnb = 0
    for i in range(len(mini_dev_labels)):
        if mnb_predict[i]==mini_dev_labels[i]:
            total_right_mnb += 1
    print "mnb ratio correct", total_right_mnb/float(len(mini_dev_labels))
    print mnb_f1'''

    print 'Training logistic regression model'
    logregr = LogisticRegression(penalty='l2', C=0.05, tol=0.01)
    logregr.fit(fe.mini_train_data, fe.mini_train_labels)
    weights = logregr.coef_
    dev_predict = logregr.predict(fe.submission_data)
    # f1_score = metrics.f1_score(mini_dev_labels, dev_predict)
    # total_right = 0
    # for i in range(len(mini_dev_labels)):
    #     if dev_predict[i]==mini_dev_labels[i]:
    #         total_right += 1

    print 'Prediction submission data'
    output_matrix = logregr.predict_proba(fe.submission_data)
    print output_matrix.shape
    print logregr.classes_
    print output_matrix[:10]
    #print "LR ratio correct", total_right/float(len(mini_dev_labels))

    print 'Generate CSV file with prediction on submission data'
    mycsvwriter = csv.writer(open('ml_watchers_submission.csv','wb'))
    for row in output_matrix:
        mycsvwriter.writerow(row)

if __name__ == '__main__':
    main()
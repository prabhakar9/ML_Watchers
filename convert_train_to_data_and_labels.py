import csv
from feature_engineering import FeatureEngineering
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
    fe.prepare_data('train.csv', 'test.csv')
    #fe.load_data('train_processed.csv', 'test_processed.csv')

    ''' + Examples of how to access the data prepared in FeatureEngineering class
    fe.train_labels, fe.train_data, fe.mini_train_data, fe.mini_label_data
    fe.test_data, fe.test_labels
    fe.submission_data -> this is the data we need to predict and submit'''

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
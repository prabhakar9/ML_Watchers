from feature_engineering import *

def main():
    fe = FeatureEngineering()
    fe.load_train_data()

    lr = LogisticRegression(C=.1, tol=0.01)
    lr.fit(fe.train_data, fe.train_labels)

    test_data_A = fe.test_data[:10000]
    test_labels_A = fe.test_labels[:10000]

    results = lr.predict(test_data_A)
    print 'f-score = %f' % metrics.f1_score(test_labels_A, results, average = 'weighted')
    print 'Accuracy = %f' % metrics.accuracy_score(test_labels_A, results)

if __name__ == '__main__':
    main()
from feature_engineering import *

def main():
    fe = FeatureEngineering()
    #fe.prepare_data('train.csv','test.csv')
    fe.load_data()

    print fe.train_data.shape
    print fe.mini_train_data.shape
    print fe.test_data.shape
    print fe.submission_data.shape

if __name__ == '__main__':
    main()
# General Libraries
import time
import numpy as np
import csv
import pandas as pd

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

# StandardScaler - used to scale numerical features
# DictVectorizer - used for all features, but it might be too slow
# OneHotEncoder - used for categorical features
# LabelBinarizer - used for labels only

class FeatureDetails:
    def __init__(self):
        self.features = []
        self.is_categorical_index = []

    def add_feature(self, name, categories = None):
        f = FeatureInfo(name, len(self.is_categorical_index), categories)
        if f.is_categorical:
            self.is_categorical_index += ([True] * len(categories))
        else:
            self.is_categorical_index += [False]
        self.features += [f]

    def print_features(self):
        for feature in self.features:
            print 'Name = %s, Start index = %s, End index = %s' % (feature.name, feature.start_index, feature.end_index)

    def create_original_array(self, numerical, categorical):
        index_numerical = 0
        index_categorical = 0
        X = []

        #for index in range(0, len(self.is_categorical_index)):
        for flag in self.is_categorical_index:
            if index_numerical == 0 and index_categorical == 0:
                if flag:
                    X = categorical[:,0:1]
                    index_categorical += 1
                else:
                    X = numerical[:,0:1]
                    index_numerical += 1
                continue

            if flag:
                X = np.hstack((X, categorical[:,index_categorical:index_categorical+1]))
                index_categorical += 1
            else:
                X = np.hstack((X, numerical[:,index_numerical:index_numerical+1]))
                index_numerical += 1

        return X

    def get_categorical_features(self):
        return np.array(self.is_categorical_index)

    def get_numerical_features(self):
        return np.array([not i for i in self.is_categorical_index])

    def get_row_info(self, list):
        return 'Test'

class FeatureInfo:
    def __init__(self, name, start_index, categories = None):
        self.name = name
        self.categories = categories
        self.is_categorical = categories != None
        self.start_index = start_index
        self.end_index = start_index
        if self.categories:
            self.end_index += len(categories) - 1

#
class FeatureEngineering:
    districts = ['NORTHERN', 'PARK', 'INGLESIDE', 'BAYVIEW', 'RICHMOND', 'CENTRAL', 'TARAVAL', 'TENDERLOIN', 'MISSION', 'SOUTHERN']
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    times_of_day = ['Twilight', 'Morning', 'Afternoon', 'Night']

    #days_of_week = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6 }
    #districts = {'RICHMOND': 6, 'CENTRAL': 1, 'NORTHERN': 4, 'TARAVAL': 8, 'BAYVIEW': 0, 'INGLESIDE': 2, 'PARK': 5, 'MISSION': 3, 'TENDERLOIN': 9, 'SOUTHERN': 7}
    #time_of_day = {'Twilight':0, 'Morning':1, 'Afternoon':2, 'Night':3}
    labels = {'RECOVERED VEHICLE': 24, 'SUICIDE': 31, 'FRAUD': 13, 'WEAPON LAWS': 38, 'ROBBERY': 25, 'ARSON': 0, 'SECONDARY CODES': 27, 'SEX OFFENSES FORCIBLE': 28, 'WARRANTS': 37, 'PROSTITUTION': 23, 'DRUG/NARCOTIC': 7, 'EMBEZZLEMENT': 9, 'TRESPASS': 34, 'LOITERING': 18, 'KIDNAPPING': 15, 'DRIVING UNDER THE INFLUENCE': 6, 'LARCENY/THEFT': 16, 'VANDALISM': 35, 'NON-CRIMINAL': 20, 'BURGLARY': 4, 'BAD CHECKS': 2, 'STOLEN PROPERTY': 30, 'EXTORTION': 10, 'SUSPICIOUS OCC': 32, 'PORNOGRAPHY/OBSCENE MAT': 22, 'LIQUOR LAWS': 17, 'FAMILY OFFENSES': 11, 'SEX OFFENSES NON FORCIBLE': 29, 'TREA': 33, 'GAMBLING': 14, 'BRIBERY': 3, 'VEHICLE THEFT': 36, 'FORGERY/COUNTERFEITING': 12, 'ASSAULT': 1, 'DRUNKENNESS': 8, 'MISSING PERSON': 19, 'DISORDERLY CONDUCT': 5, 'OTHER OFFENSES': 21, 'RUNAWAY': 26}
    train_data, train_labels, mini_train_data, mini_label_data, dev_data, dev_labels, test_data, test_labels = None, None, None, None, None, None, None, None
    submission_data = None
    db = None # Remove
    features = None
    scalar = None
    pca = None

    # We should add any code we might need to initialize
    def __init__(self):
        print 'Initializing'

        self.features = FeatureDetails()
        self.features.add_feature('Year')
        self.features.add_feature('Month')
        self.features.add_feature('Day')
        self.features.add_feature('Hour')
        self.features.add_feature('Minute')
        self.features.add_feature('Time of Day', self.times_of_day)
        self.features.add_feature('Day of Week', self.days_of_week)
        self.features.add_feature('District', self.districts)
        self.features.add_feature('Coordinate X')
        self.features.add_feature('Coordinate Y')
        self.features.add_feature('Block')

        self.scaler = StandardScaler()
        self.pca = PCA(2)

        #self.load_data_2('mini_train.csv', True)
        self.load_data_2('train.csv', True)
        self.load_data_2('test.csv', False)

    def load_data_2(self, csv_file, is_training_data):
        # Read all train data and convert categorical features into ints
        print 'Extracting %s data' % ('training' if is_training_data else 'test')
        X, Y = self._extract_data(csv_file, is_training_data)

        # Split into 2 arrays: numerical and categorical
        categorical = self.features.get_categorical_features()
        numerical = self.features.get_numerical_features()

        print X.shape
        print len(categorical)
        print len(numerical)
        print sum(categorical)
        print sum(numerical)

        # For numerical, use StandardScaler to scale data
        if is_training_data:
            X_num = self.scaler.fit_transform(X[:,numerical])
        else:
            X_num = self.scaler.transform(X[:,numerical])

            # Save mean and std for each numerical feature so we can change back later if we want to....
            #print scaler.inverse_transform(X2[0])
            #print scaler.mean_
            #print scaler.std

        # For categorical leave as is
        X_cat = X[:,categorical]

        # Convert back into a single ndarray_
        X_new = self.features.create_original_array(X_num, X_cat)

        print X_new.shape

        # TODO: Ability to add new features that are derived: PCA, combining, clustering, etc.
        # Add new 2 dimensional PCA vector
        '''if is_training_data:
            X4 = self.pca.fit_transform(X_new)
        else:
            X4 = self.pca.transform(X_new)
        print X4.shape
        print X_new.shape
        #print X_new[12]
        X_new = np.hstack((X_new, X4))
        print X_new.shape
        #print X_new[12]'''


    # Loads all the data that was previously prepared by the prepare_data() method.
    def load_data(self, train_csv, test_csv):
        X, Y = [], []
        with open(train_csv, 'r') as f:
            for line in csv.reader(f):
                X += [line[:-1]]
                Y += [line[-1]]

        # Shuffle the data and divide it into train, mini_train, dev and test sets.
        self._shuffle_and_set_datasets(X, Y)

        submission_data = []
        with open(test_csv, 'r') as f:
            for line in csv.reader(f):
                submission_data += [line]

        self.submission_data = np.asarray(submission_data)

        # Report status
        print ' + Status report:'
        print '    - Train data and labels: %s - %s' % (self.train_data.shape, self.train_labels.shape)
        print '    - Mini train data and labels: %s - %s' % (self.mini_train_data.shape, self.mini_train_labels.shape)
        print '    - Dev data and labels: %s - %s' % (self.dev_data.shape, self.dev_labels.shape)
        print '    - Test data and labels: %s - %s' % (self.test_data.shape, self.test_labels.shape)
        print '    - Submission data:', self.submission_data.shape

    # Method used to prepare data for the first time and perform all the required feature engineering. In subsequent
    # runs you only need to run the load_data() method.
    def prepare_data(self, train_csv, test_csv, load_data = False):
        print 'Extracting training data'
        X, Y = self._extract_data(train_csv, True)

        print 'Extracting submission data that our model will need to predict'
        self.submission_data, _ = self._extract_data(test_csv, False)

        # Create dummy variables for Street1 and Street2 variables and remove them from the X array.
        df = pd.DataFrame(np.vstack((X, self.submission_data))[:,-2:], columns = ['Street1', 'Street2'])

        print 'Creating dummy variables for [Street 1] feature'
        dummy_street1 = pd.get_dummies(df['Street1'])
        print 'Creating dummy variables for [Street 2] feature'
        dummy_street2 = pd.get_dummies(df['Street2'])

        print 'Add [Street 1] dummy variables to feature vector'
        X = np.hstack((X[:,:-2], dummy_street1.values[:len(X),:]))
        print 'Add [Street 2] dummy variables to feature vector'
        X = np.hstack((X, dummy_street2.values[:len(X),:]))

        print 'Add [Street 1] dummy variables to submission feature vector'
        self.submission_data = np.hstack((self.submission_data[:,:-2], dummy_street1.values[len(X):,:]))
        print 'Add [St reet 2] dummy variables to submission feature vector'
        self.submission_data = np.hstack((self.submission_data, dummy_street2.values[len(X):,:]))

        print 'Save all this data in CSV files'
        train_csv = 'train_processed.csv'
        test_csv = 'test_processed.csv'
        np.savetxt(train_csv, np.hstack((X, np.reshape(Y, (200,1)))), delimiter=',', fmt='%s')
        np.savetxt(test_csv, self.submission_data, delimiter=',', fmt='%s')

        if load_data:
            print 'Load data that was just prepared'
            self.load_data(train_csv, test_csv)

    # Internal method used to extract data from CSV file and vectorizing categorical features.
    def _extract_data(self, csv_path, is_training_data):
        X, Y = [], []
        print 'Reading %s data' % ('training' if is_training_data else 'test')
        data = np.array(pd.read_csv(csv_path))

        #
        if is_training_data:
            print 'Extracting unique Street 1 and 2 values'
            street_values = []
            for line in data:
                street_values += [self._get_address_values(line[6])[-2:]]
            street_values = np.array(street_values)

            print 'Getting unique Street 1 values'
            self.streets_1 = np.unique(street_values[:,0]).tolist()
            self.features.add_feature('Streets 1', self.streets_1)

            print 'Getting unique Street 2 values'
            self.streets_2 = np.unique(street_values[:,1]).tolist()
            self.features.add_feature('Streets 2', self.streets_2)

            # Dispose since it's no longer needed
            street_values = []

        print 'Starting to process all rows'
        count = 0
        for line in data:
            try:
                feature_values = []

                if is_training_data:
                    date = line[0]
                    category = line[1]
                    day_of_week = line[3]
                    district = line[4]
                    address = line[6]
                    x = line[7]
                    y = line[8]

                    Y += [self.labels[category]]
                else:
                    id = line[0]
                    date = line[1]
                    day_of_week = line[2]
                    district = line[3]
                    address = line[4]
                    x = line[5]
                    y = line[6]

                # Extract all feature values from this line
                feature_values += self._extract_datetime_features(date, is_training_data)
                feature_values += [1 if day_of_week == d else 0 for d in self.days_of_week]
                feature_values += [1 if district == d else 0 for d in self.districts]
                feature_values += [float(x), float(y)]
                feature_values += self._get_address_values(address, convert_to_dummy_variables = True)

                X += [feature_values]

                count += 1
                if count % 50000 == 0: print ' >>> ' + str(count)
                if count > 100000:
                    print 'Done with 100K rows'
                    break
            except Exception as inst:
                print 'Failed to process row. Exception: ', inst

        '''
        # Open csv file and read all lines.
        with open(csv_path, 'r') as f:
            counter = -1

            # Go through all lines and process them
            for line in csv.reader(f):
                # Use this to skip header row
                counter += 1
                if counter == 0: continue

                try:
                    feature_values = []

                    if is_training_data:
                        date = line[0]
                        category = line[1]
                        day_of_week = line[3]
                        district = line[4]
                        address = line[6]
                        x = line[7]
                        y = line[8]

                        Y += [self.labels[category]]
                    else:
                        id = line[0]
                        date = line[1]
                        day_of_week = line[2]
                        district = line[3]
                        address = line[4]
                        x = line[5]
                        y = line[6]

                    # Extract all feature values from this line
                    feature_values += self._extract_datetime_features(date, is_training_data)
                    feature_values += [1 if day_of_week == d else 0 for d in self.days_of_week]
                    feature_values += [1 if district == d else 0 for d in self.districts]
                    feature_values += [float(x), float(y)]
                    feature_values += self._get_address_values(address)

                    X += [feature_values]
                except Exception as inst:
                    print 'Failed to process row. Exception: ', inst'''

        # Return both X and Y values. Note that Y will be empty if the data is unlabeled.
        return (np.asarray(X), np.asarray(Y))

    # Shuffle and set train, mini_train, dev, and test datasets.
    def _shuffle_and_set_datasets(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)

        # Initialize in variables in case the method is called more than once.
        np.random.seed(1023709456)

        # Shuffle the data and divide it into train, mini_train, dev and test sets.
        shuffle = np.random.permutation(np.arange(X.shape[0]))
        X, Y = X[shuffle], Y[shuffle]
        self.train_data, self.train_labels = X[:580000], Y[:580000]
        self.mini_train_data, self.mini_train_labels = X[:10000], Y[:10000]
        self.dev_data, self.dev_labels = X[580000:730000], Y[580000:730000]
        self.test_data, self.test_labels = X[730000:], Y[730000:]

    # Normalize date and convert it into a list of 5 elements: year, month, day, hour, minute.
    def _extract_datetime_features(self, date, is_training_data):
        format = "%Y-%m-%d %H:%M:%S" if is_training_data == True else "%m/%d/%Y %H:%M"
        #format = "%Y-%m-%d %H:%M:%S" #if is_labeled_data == True else "%m/%d/%Y %H:%M"

        time_obj = time.strptime(date, format)
        output = []
        output += [time_obj.tm_year] # Year
        output += [time_obj.tm_mon] # Month
        output += [time_obj.tm_mday] # Day
        output += [time_obj.tm_hour] # Hour
        output += [(time_obj.tm_hour * 60) + time_obj.tm_min] # Minute of Day

        # Add Time of Day as categorical variable.
        time_of_day = self._get_time_of_day(time_obj.tm_hour)
        output += [1 if time_of_day == t else 0 for t in self.times_of_day]

        return output

    # Convert hour into categorical variable with the following values: Night, Morning, Daytime, Evening.
    def _get_time_of_day(self, hour):
        if hour < 6:
            return 'Twilight'
        elif hour < 12:
            return 'Morning'
        elif hour < 18:
            return 'Afternoon'
        else:
            return 'Night'

    # Convert hour into categorical variable with the following values: Night, Morning, Daytime, Evening.
    # These are in turn converted to dummy variables.
    def _get_time_of_day_OLD(self, hour):
        # Night
        if hour < 6:
            return [1, 0, 0, 0]
        # Morning
        elif hour < 12:
            return [0, 1, 0, 0]
        # Daytime
        elif hour < 18:
            return [0, 0, 1, 0]
        # Evening
        else:
            return [0, 0, 0, 1]

    def _get_day_of_week(self, day_of_week):
        return [1 if day_of_week == d else 0 for d in self.days_of_week]

    # Converts address into following columns:
    #    + Block number - if not existent then it's 0
    #    + Street 1 - always available
    #    + Street 2 - not available where address is of "block" type
    # NOTE: These are not converted into dummy variables, that will be done afterwards.
    def _get_address_values(self, address, convert_to_dummy_variables = False):
        address = address.lower()
        block = 0
        street_1 = ''
        street_2 = ''

        if '/' in address:
            list = address.split('/')
            street_1 = list[0].strip()
            street_2 = list[1].strip()
        elif ' of ' in address:
            list = address.split(' of ')
            block = int(list[0].split('block')[0].strip())
            street_1 = list[1].strip()

        if convert_to_dummy_variables:
            list = [block]
            list += [1 if street_1 == s else 0 for s in self.streets_1]
            list += [1 if street_2 == s else 0 for s in self.streets_2]
            return list
        else:
            return [block, street_1, street_2]

def main():
    print 'Hello'

if __name__ == '__main__':
    main()


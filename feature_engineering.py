import time
import numpy as np
import csv

class FeatureEngineering:
    days_of_week = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6 }
    districts = {'RICHMOND': 6, 'CENTRAL': 1, 'NORTHERN': 4, 'TARAVAL': 8, 'BAYVIEW': 0, 'INGLESIDE': 2, 'PARK': 5, 'MISSION': 3, 'TENDERLOIN': 9, 'SOUTHERN': 7}
    labels = {'RECOVERED VEHICLE': 24, 'SUICIDE': 31, 'FRAUD': 13, 'WEAPON LAWS': 38, 'ROBBERY': 25, 'ARSON': 0, 'SECONDARY CODES': 27, 'SEX OFFENSES FORCIBLE': 28, 'WARRANTS': 37, 'PROSTITUTION': 23, 'DRUG/NARCOTIC': 7, 'EMBEZZLEMENT': 9, 'TRESPASS': 34, 'LOITERING': 18, 'KIDNAPPING': 15, 'DRIVING UNDER THE INFLUENCE': 6, 'LARCENY/THEFT': 16, 'VANDALISM': 35, 'NON-CRIMINAL': 20, 'BURGLARY': 4, 'BAD CHECKS': 2, 'STOLEN PROPERTY': 30, 'EXTORTION': 10, 'SUSPICIOUS OCC': 32, 'PORNOGRAPHY/OBSCENE MAT': 22, 'LIQUOR LAWS': 17, 'FAMILY OFFENSES': 11, 'SEX OFFENSES NON FORCIBLE': 29, 'TREA': 33, 'GAMBLING': 14, 'BRIBERY': 3, 'VEHICLE THEFT': 36, 'FORGERY/COUNTERFEITING': 12, 'ASSAULT': 1, 'DRUNKENNESS': 8, 'MISSING PERSON': 19, 'DISORDERLY CONDUCT': 5, 'OTHER OFFENSES': 21, 'RUNAWAY': 26}
    train_data, train_labels, mini_train_data, mini_label_data, dev_data, dev_labels, test_data, test_labels = None, None, None, None, None, None, None, None
    submission_data = None

    # We should add any code we might need to initialize
    def __init__(self):
        print 'Initializing'

    # Main method used to prepare data
    def prepare_data(self, train_csv, test_csv):
        # Initialize in variables in case the method is called more than once.
        np.random.seed(1023709456)

        print 'Extracting training data'
        X, Y = self._extract_data(train_csv, True)

        # Shuffle the data and divide it into train, mini_train, dev and test sets.
        shuffle = np.random.permutation(np.arange(X.shape[0]))
        X, Y = X[shuffle], Y[shuffle]
        self.train_data, self.train_labels = X[:840000], Y[:840000]
        self.mini_train_data, self.mini_train_labels = X[:10000], Y[:10000]
        self.dev_data, self.dev_labels = X[840000:860000], Y[840000:860000]
        self.test_data, self.test_labels = X[860000:], Y[860000:]

        print 'Extracting submission data that our model will need to predict'
        self.submission_data, Y = self._extract_data(test_csv, False)

        # Report status (remove later)
        print ' + Status report:'
        print '    - Train data and labels: %s - %s' % (self.train_data.shape, self.train_labels.shape)
        print '    - Mini train data and labels: %s - %s' % (self.mini_train_data.shape, self.mini_train_labels.shape)
        print '    - Dev data and labels: %s - %s' % (self.dev_data.shape, self.dev_labels.shape)
        print '    - Test data and labels: %s - %s' % (self.test_data.shape, self.test_labels.shape)
        print '    - Submission data:', self.submission_data.shape

    # Internal method used to extract data from CSV file and convert feature vector
    def _extract_data(self, csv_path, is_labeled_data):
        X, Y = [], []

        # Open csv file and read all lines.
        with open(csv_path, 'r') as f:
            counter = -1
            reader = csv.reader(f)

            # Go through all lines and process them
            for line in reader:
                # Use this to skip header row
                counter += 1
                if counter == 0:
                    continue

                try:
                    feature_values = []

                    if is_labeled_data == True:
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
                    feature_values += self._normalize_date(date, is_labeled_data)
                    feature_values += [self.days_of_week[day_of_week]]
                    feature_values += [self.districts[district]]
                    feature_values += self._normalize_coordinates(x, y)

                    X += [feature_values]
                except Exception as inst:
                    print 'Failed to process row. Exception: ', inst

        # Return both X and Y values. Note that Y will be empty if the data is unlabeled.
        return (np.asarray(X), np.asarray(Y))

    # Normalize date and convert it into a list of 5 elements: year, month, day, hour, minute.
    @staticmethod
    def _normalize_date(date, is_labeled_data):
        format = "%Y-%m-%d %H:%M:%S" if is_labeled_data == True else "%m/%d/%Y %H:%M"

        time_obj = time.strptime(date, format)
        output = []
        output += [time_obj.tm_year - 2012]
        output += [time_obj.tm_mon]
        output += [time_obj.tm_mday]
        output += [time_obj.tm_hour]
        output += [time_obj.tm_min]
        return output

    # Normalize coordinates based on mean and standard deviation of both x and y coordinates.
    @staticmethod
    def _normalize_coordinates(x, y):
        # TODO: Let's just make sure these are accurate :)
        x_mean =  -122.422042526
        x_std =  0.0244209517397
        y_mean =  37.7605806147
        y_std =  0.0260142906993

        x = float(x)
        y = float(y)
        return [(x - x_mean)/x_std, (y - y_mean)/y_std]
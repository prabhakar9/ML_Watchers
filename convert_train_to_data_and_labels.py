import time
import numpy as np
import sys
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



days_of_week = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6 }
districts = {'RICHMOND': 6, 'CENTRAL': 1, 'NORTHERN': 4, 'TARAVAL': 8, 'BAYVIEW': 0, 'INGLESIDE': 2, 'PARK': 5, 'MISSION': 3, 'TENDERLOIN': 9, 'SOUTHERN': 7}
labels = {'RECOVERED VEHICLE': 24, 'SUICIDE': 31, 'FRAUD': 13, 'WEAPON LAWS': 38, 'ROBBERY': 25, 'ARSON': 0, 'SECONDARY CODES': 27, 'SEX OFFENSES FORCIBLE': 28, 'WARRANTS': 37, 'PROSTITUTION': 23, 'DRUG/NARCOTIC': 7, 'EMBEZZLEMENT': 9, 'TRESPASS': 34, 'LOITERING': 18, 'KIDNAPPING': 15, 'DRIVING UNDER THE INFLUENCE': 6, 'LARCENY/THEFT': 16, 'VANDALISM': 35, 'NON-CRIMINAL': 20, 'BURGLARY': 4, 'BAD CHECKS': 2, 'STOLEN PROPERTY': 30, 'EXTORTION': 10, 'SUSPICIOUS OCC': 32, 'PORNOGRAPHY/OBSCENE MAT': 22, 'LIQUOR LAWS': 17, 'FAMILY OFFENSES': 11, 'SEX OFFENSES NON FORCIBLE': 29, 'TREA': 33, 'GAMBLING': 14, 'BRIBERY': 3, 'VEHICLE THEFT': 36, 'FORGERY/COUNTERFEITING': 12, 'ASSAULT': 1, 'DRUNKENNESS': 8, 'MISSING PERSON': 19, 'DISORDERLY CONDUCT': 5, 'OTHER OFFENSES': 21, 'RUNAWAY': 26}
x_mean =  -122.422042526
x_std =  0.0244209517397
y_mean =  37.7605806147
y_std =  0.0260142906993
# create global data structure:
# columns: year, month, day, hour, minute, dayofweek, district, x-coord, y-coord
# output_component [0, 0, 0, 0, 0, 0, 0, 0.0, 0.0]
output_list = []
label_list = []
#x_list = []
#y_list = []


def date_worker(date, output_component):
    try:
        time_obj = time.strptime(date, "%Y-%m-%d %H:%M:%S")
        output_component[0] = time_obj.tm_year - 2012
        output_component[1] = time_obj.tm_mon
        output_component[2] = time_obj.tm_mday
        output_component[3] = time_obj.tm_hour
        output_component[4] = time_obj.tm_min
    except:
        print "date is:", date
    # print time_obj.tm_year, time_obj.tm_mon, time_obj.tm_mday, time_obj.tm_min
    return output_component

def dayofweek_district_worker(line, output_component):
    for i in range(3,9):
        if line[i] in days_of_week:
            output_component[5] = days_of_week[line[i]]
            output_component[6] = districts[line[i+1]]
            break
    return output_component

def label_worker(line, label_list):
    item = labels[line[1]]
    label_list.append(item)
    return label_list

def coordinate_worker(line, output_component, counter):
    x = float(line[-2])
    # print "x", x
    y = float(line[-1])
    output_component[-2] = (x - x_mean)/x_std
    output_component[-1] = (y - y_mean)/y_std

    return output_component

with open("../train.csv", 'r') as f:
    counter = 0

    for line in f:
        output_component = [0, 0, 0, 0, 0, 0, 0, 0.0, 0.0]
        line = line.rstrip()
        line = line.split(',')
        # print len(line)
        if counter > 0:

            date = line[0]
            output_component = date_worker(date, output_component)
            output_component = dayofweek_district_worker(line, output_component)
            output_component = coordinate_worker(line, output_component, counter)
            label_list = label_worker(line, label_list)

            output_list.append(output_component)

        counter += 1


all_data = np.asarray(output_list)
all_label = np.asarray(label_list, int)

shuffle = np.random.permutation(np.arange(all_data.shape[0]))
X, Y = all_data[shuffle], all_label[shuffle]

print "shape", X.shape, Y.shape

dev_data, dev_labels = X[500000:878049], Y[500000:878049]
train_data, train_labels = X[:500000], Y[:500000]
mini_train_data, mini_train_labels = X[:10000], Y[:10000]
mini_dev_data, mini_dev_labels = X[500000:510000], Y[500000:510000]

# can't use MNB yet - need non-negative integer data
'''

mnb = MultinomialNB(alpha = 0.01)
mnb.fit(mini_train_data, mini_train_labels)
mnb_predict = mnb.predict(mini_dev_data)
mnb_f1 = metrics.f1_score(mini_dev_labels, mnb_predict)
total_right_mnb = 0
for i in range(len(mini_dev_labels)):
    if mnb_predict[i]==mini_dev_labels[i]:
        total_right_mnb += 1
print "mnb ratio correct", total_right_mnb/float(len(mini_dev_labels))
print mnb_f1
'''

logregr = LogisticRegression(penalty='l2', C=0.05)
logregr.fit(mini_train_data, mini_train_labels)
weights = logregr.coef_
dev_predict = logregr.predict(mini_dev_data)
# f1_score = metrics.f1_score(mini_dev_labels, dev_predict)
total_right = 0
for i in range(len(mini_dev_labels)):
    if dev_predict[i]==mini_dev_labels[i]:
        total_right += 1
print "LR ratio correct", total_right/float(len(mini_dev_labels))

# print f1_score


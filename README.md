# ML_Watchers Summary 

## Overview of models

|  Model | Accuracy | Train Time | Kaggle Score | Notes |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 31% | few minutes| 84th |see code at ...|
| Logistic Regression | ~30%| few minutes | 105th |https://github.com/prabhakar9/ML_Watchers/blob/master/convert_train_to_data_and_labels.py|

## Initial Features

These are the features we get in both training and test data sets.

| Date | DayOfWeek | PdDistrict | Address | X | Y |
|---|---|---|---|---|---|---|---|
| 2015-05-10 23:59:00 | Sunday | BAYVIEW | 2000 Block of THOMAS AV | -122.3995877042 | 37.7350510104 |

## Feature Engineering

Those initial features are converted into the features described below. The first 6 are based on **Date**, while the last 3 are based on **Address** (more details in sections below).
 
1. **Year**
2. **Month** - Value between 1-12
3. **Day of Month** - Value between 1-31
4. **Hour** - Value between 0-23
5. **Minute of Day** - Value between 0-1439. Calculated by the following formula: ```(hour * 60) + minute``` 
6. **Time of Day** - Categorical feature with the following possible values: Twilight (12 am to 6 am), Morning (6 am to 12 pm), Afternoon (12 pm to 6 pm), Night (6 pm to 12 am)
7. **Day of Week** - Categorical feature with 7 distinct values: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.
8. **District** - Categorical feature with 10 dictinct values: NORTHERN, PARK, INGLESIDE, BAYVIEW, RICHMOND, CENTRAL, TARAVAL, TENDERLOIN, MISSION, SOUTHERN.
9. **X** - X coordinates as float
10. **Y** - Y coordinates as float
11. **Block** - Categorical variable containing the block number (if not available it's assigned default/empty category). There are 85 distinct values.
12. **Street 1** - Categorical variable containing the first street/avenue/etc. There are 2033 distinct values.
13. **Street 2** - Categorical variable containing the second street/avenue/etc (if not available it's assigned default/empty category). There are 1694 distinct values. 
 
The categorical features, with the exception of the last 3 which we'll discuss later, are converted to dummy variables. That causes the number of features to expand based on the number of distinct values. For example, the **Time of Day** feature is converted to the following 4 features, each with possible values of 0 or 1:

* **IsTwilight**
* **IsMorning**
* **IsAfternoon**
* **IsNight**

As for the features based on **Address** (the last 3), they contained too many distinct values which made it impractical to use with our computers. As such, we first converted each one to a int value (Category1 = 0, Category2, 1, etc.) and then we used the sklearn.preprocessing.OneHotEncoder to convert all of them together into a sparse matrix which made it possible to hold in memory. Afterwards, we reduced the dimensionality to 20 columns using a truncated SVD (aka LSA). We managed to retain more than 44% of the variance.

Finally, we used the sklearn.preprocessing.StandardScaler to do feature scaling on all the numerical features.

## Final set of Features 

These are the features we were left with after feature engineering has been done:

![alt text](/features.jpg "Features")

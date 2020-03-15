import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load the game dataset
games=pd.read_csv("C:\\Users\\hshah\\Desktop\\"
                  "building machine learning project with python\\"
                  "Board game review prediction\\games.csv")

# print(games.columns)
#
# print(games.shape)
#
# plt.hist(games["average_rating"])
# plt.show()

# print the first row of all the games with zero scores
print(games[games["average_rating"] == 0].iloc[0])

# print the first row of all the games with score greater then zero
print(games[games["average_rating"] > 0].iloc[0])

# Remove any rows without user reviews
games=games[games["users_rated"] > 0]

# Remove any rows with missing values
games=games.dropna(axis=0)

# Make a histogram of all the average ratings
# plt.hist(games["average_rating"])
# plt.show()

print(games.columns)

# Correlation matrix

# corrmat = games.corr()
# fig = plt.figure(figsize=(12,9))
#
# sns.heatmap(corrmat, vmax= .8, square= True)
# plt.show()

# get all the coumns from the dataframe
columns=games.columns.tolist()

# filter the columns that we do not want
columns=[c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

# store the variable we wil be predicting on
target="average_rating"

# generate the training and test datasets

# generate training set
train=games.sample(frac=0.8, random_state=1)

# Select anything not in the training set and put it in test
test=games.loc[~games.index.isin(train.index)]

# print shapes
print(train.shape)
print(test.shape)

# initialize the model class
LR=LinearRegression()

# fit the model of the training data
LR.fit(train[columns], train[target])

# generate predictions for the test dataset
predictions=LR.predict(test[columns])

# compute error between our test predictions and actual values
print(mean_squared_error(predictions, test[target]))

# initialize the model
RFR=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)

# fit to the data
RFR.fit(train[columns], train[target])

# make predictions
predictions=RFR.predict(test[columns])

# compute the error between our test predictions and actual values
print(mean_squared_error(predictions, test[target]))

# Make predictions with both the models
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1, -1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1, -1))

# print out the predictions
print(rating_LR)
print(rating_RFR)
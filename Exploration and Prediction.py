# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:05:29 2017

@author: sal
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, date, time, timedelta
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge



df = pd.read_csv('C:/Users/HP/Desktop/datanonzero.csv')


# getting descriptive statistics of dataframe

print df.describe()


# getting scatter matrix

sns.pairplot(df)
plt.show()


# saving the column values in variables for later use

android_rating_1 = df['android_rating_1']
android_rating_2 = df['android_rating_2']
android_rating_3 = df['android_rating_3']
android_rating_4 = df['android_rating_4']
android_rating_5 = df['android_rating_5']
android_total_ratings = df['android_total_ratings']
ios_all_ratings = df['ios_all_ratings']
dates = df['dates']
ios_current_ratings=df['ios_current_ratings']



# Getting pearson's corr coeffs for the pairs with high correlations


print("\n Pearson correlation coeffecients for identified pairs")
print 65*'_'


print("\n The pearson corr coeff between android_total_ratings and ios_all_ratings is {}".format(np.corrcoef(ios_all_ratings,android_total_ratings)[0][1]))
print("\n The pearson corr coeff between android_total_ratings and android_rating_1 is {}".format(np.corrcoef(android_rating_1,android_total_ratings)[0][1]))
print(" The pearson corr coeff between android_total_ratings and android_rating_2 is {}".format(np.corrcoef(android_rating_2,android_total_ratings)[0][1]))
print(" The pearson corr coeff between android_total_ratings and android_rating_3 is {}".format(np.corrcoef(android_rating_3,android_total_ratings)[0][1]))
print(" The pearson corr coeff between android_total_ratings and android_rating_4 is {}".format(np.corrcoef(android_rating_4,android_total_ratings)[0][1]))
print(" The pearson corr coeff between android_total_ratings and android_rating_5 is {}".format(np.corrcoef(android_rating_5,android_total_ratings)[0][1]))
print(" \n The pearson corr coeff between android_rating_1 and ios_all_ratings is {}".format(np.corrcoef(ios_all_ratings,android_rating_1)[0][1]))
print(" The pearson corr coeff between android_rating_2 and ios_all_ratings is {}".format(np.corrcoef(ios_all_ratings,android_rating_2)[0][1]))
print(" The pearson corr coeff between android_rating_3 and ios_all_ratings is {}".format(np.corrcoef(ios_all_ratings,android_rating_3)[0][1]))
print(" The pearson corr coeff between android_rating_4 and ios_all_ratings is {}".format(np.corrcoef(ios_all_ratings,android_rating_4)[0][1]))
print(" The pearson corr coeff between android_rating_5 and ios_all_ratings is {}".format(np.corrcoef(ios_all_ratings,android_rating_5)[0][1]))
print
print("Checking for multicollinearity")
print 65*'_'
print(" The pearson corr coeff between android_rating_2 and android_rating_1 is {}".format(np.corrcoef(android_rating_2,android_rating_1)[0][1]))
print(" The pearson corr coeff between android_rating_3 and android_rating_1 is {}".format(np.corrcoef(android_rating_3,android_rating_1)[0][1]))
print(" The pearson corr coeff between android_rating_4 and android_rating_1 is {}".format(np.corrcoef(android_rating_4,android_rating_1)[0][1]))
print(" The pearson corr coeff between android_rating_5 and android_rating_1 is {}".format(np.corrcoef(android_rating_5,android_rating_1)[0][1]))


print 65*'_'
print(" The pearson corr coeff between android_rating_1 and android_rating_2 is {}".format(np.corrcoef(android_rating_1,android_rating_2)[0][1]))
print(" The pearson corr coeff between android_rating_3 and android_rating_2 is {}".format(np.corrcoef(android_rating_3,android_rating_2)[0][1]))
print(" The pearson corr coeff between android_rating_4 and android_rating_2 is {}".format(np.corrcoef(android_rating_4,android_rating_2)[0][1]))
print(" The pearson corr coeff between android_rating_5 and android_rating_2 is {}".format(np.corrcoef(android_rating_5,android_rating_2)[0][1]))


print 65*'_'
print(" The pearson corr coeff between android_rating_1 and android_rating_3 is {}".format(np.corrcoef(android_rating_1,android_rating_3)[0][1]))
print(" The pearson corr coeff between android_rating_2 and android_rating_3 is {}".format(np.corrcoef(android_rating_2,android_rating_3)[0][1]))
print(" The pearson corr coeff between android_rating_4 and android_rating_3 is {}".format(np.corrcoef(android_rating_4,android_rating_3)[0][1]))
print(" The pearson corr coeff between android_rating_5 and android_rating_3 is {}".format(np.corrcoef(android_rating_5,android_rating_3)[0][1]))


print 65*'_'
print(" The pearson corr coeff between android_rating_1 and android_rating_4 is {}".format(np.corrcoef(android_rating_1,android_rating_4)[0][1]))
print(" The pearson corr coeff between android_rating_2 and android_rating_4 is {}".format(np.corrcoef(android_rating_2,android_rating_4)[0][1]))
print(" The pearson corr coeff between android_rating_3 and android_rating_4 is {}".format(np.corrcoef(android_rating_3,android_rating_4)[0][1]))
print(" The pearson corr coeff between android_rating_5 and android_rating_4 is {}".format(np.corrcoef(android_rating_5,android_rating_4)[0][1]))


print 65*'_'
print(" The pearson corr coeff between android_rating_1 and android_rating_5 is {}".format(np.corrcoef(android_rating_1,android_rating_5)[0][1]))
print(" The pearson corr coeff between android_rating_2 and android_rating_5 is {}".format(np.corrcoef(android_rating_2,android_rating_5)[0][1]))
print(" The pearson corr coeff between android_rating_3 and android_rating_5 is {}".format(np.corrcoef(android_rating_3,android_rating_5)[0][1]))
print(" The pearson corr coeff between android_rating_4 and android_rating_5 is {}".format(np.corrcoef(android_rating_4,android_rating_5)[0][1]))

"""


# plotting time series graphs 

start= datetime(2016,7,21,0,0)
end= datetime(2016,11,1,0,0)
time_delta=timedelta(minutes=10)

x=[]
while start<end:
    x.append(start)
    start=start+time_delta

x_axis=np.array(x)
t=np.array(dates)
y = np.array(android_total_ratings)
print x_axis.shape
print y.shape
print t.shape
print len(x)
fig, ax = plt.subplots(1)
fig.autofmt_xdate()
plt.plot(t,y)
xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
ax.xaxis.set_major_formatter(xfmt)
plt.show()

#plt.plot(x,y)

times = pd.date_range('2016-10-06', periods=500, freq='10min')
fig, ax = plt.subplots(1)
fig.autofmt_xdate()
plt.plot(times, range(times.size))
xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
ax.xaxis.set_major_formatter(xfmt)

plt.show()
"""


# getting prediction model for android

print("\n simple linear regression for android_total_ratings")
print 65*'_'
model1 = LinearRegression(normalize=True)
print('\n')
print(model1)
x= np.array(android_rating_5)
y= np.array(android_total_ratings)


X = x[:, None] # Transposing

#print(X)
#print(y) 

#print X.shape

model1.fit(X, y)
print('\nthe model coefficient for android model is {}'.format(model1.coef_))
print('the model intercept for android model is {}'.format(model1.intercept_))
#print (model1.residues_)
print ("The model's prediction for android_all_ratings for the given feature values is {} ".format(model1.predict([[726597],[732956],[4237996], [4239138]]))) # for the first and last value in the csv file
print("The R squared value for the model is {}".format(model1.score(X,y)))




# getting prediction model for ios

print("\n simple linear regression for ios_all_ratings")
print 65*'_'
model2 = LinearRegression(normalize=True)
print('\n')
print(model2)
x= np.array(android_rating_5)
y= np.array(ios_all_ratings)


X = x[:, None] # Transposing
#print(X)
#print(y) 
#print X.shape

model2.fit(X, y)
print('\nthe model coefficient for ios model is {}'.format(model2.coef_))
print('the model intercept for ios model is {}'.format(model2.intercept_))
#print (model2.residues_)
print ("The model's prediction for ios_all_ratings for the given feature values is {} ".format(model2.predict([[726597], [4239138]]))) # for the first and last value in the csv file
print("The R squared value for the model is {}".format(model2.score(X,y)))
print




#  multiple linear regression for android total ratings
print("\n multiple linear regression for android_total_ratings")
print 65*'_'
feature_cols = ['android_rating_5', 'android_rating_4','android_rating_3','android_rating_2']
X = df[feature_cols]
y = df.android_total_ratings
lm = LinearRegression()

# cross validation for android model

print("\nCross validation for android model")
print 65*'_'
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
lm.fit(Xtrain, ytrain)
ypred = lm.predict(Xtest)
print('the model intercept for android multiple linear regression model is {}'.format(lm.intercept_))
print('\nthe model coefficient for android multple linear regression is {}'.format(lm.coef_))
print ("The model's prediction for android_total_ratings for the given feature values is {} ".format(ypred)) 
print("The R squared value for the model is {}".format(lm.score(Xtrain,ytrain)))
print
print("The root mean sqaured error is {}".format(sqrt(mean_squared_error(ytest, ypred))))




# Trying multiple linear regression for ios total ratings
print("\n multiple linear regression for ios_all_ratings")
print 65*'_'
feature_cols = ['android_rating_5', 'android_rating_4','android_rating_3','android_rating_2']
X = df[feature_cols]
y = df.ios_all_ratings

lm = LinearRegression()
#lm.fit(X, y)



# cross validation on ios model

print("\nCross validation for android model")
print 65*'_'

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
lm.fit(Xtrain, ytrain)
ypred = lm.predict(Xtest)
print('the model intercept for ios multiple linear regression model is {}'.format(lm.intercept_))
print('\nthe model coefficient for ios multple linear regression is {}'.format(lm.coef_))
print ("The model's prediction for ios_all_ratings for the given feature values is {} ".format(ypred)) 
print("The R squared value for the model is {}".format(lm.score(Xtrain,ytrain)))
print()
print("The root mean sqaured error is {}".format(sqrt(mean_squared_error(ytest, ypred))))



# regression using statsmodel

print("\n\nRegression using statsmodel")
print 65*'_'

result = smf.ols(formula='android_total_ratings ~ android_rating_5 +android_rating_5+ ios_all_ratings   ' , data=df).fit()
print
print result.params
print result.summary()
# print the coefficients 4135326	6637612	224252

# our prediction
print
print("\n OUR PREDICTIONS")
print 65*'_'
print
print(" IOS MODEL")
print 65*'_'
feature_cols = ['android_rating_5']
X = df[feature_cols]
y = df.ios_all_ratings

lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print ("\nThe intercept is : {}".format(lm.intercept_))
print ("\nThe coeffecients are: {}".format(zip(feature_cols,lm.coef_)))
print (" \nOur prediction for the ios all rating is : {}".format(lm.predict([4239138])))

print
print
print(" ANDROID MODEL")
print 65*'_'
feature_cols = ['android_rating_5','ios_all_ratings','android_rating_1','android_rating_3']
X = df[feature_cols]
y = df.android_total_ratings

lm = LinearRegression()
lm.fit(X, y)

print ("\nThe intercept is : {}".format(lm.intercept_))
print ("\nThe coeffecients are: {}".format(zip(feature_cols,lm.coef_)))
print (" \nOur prediction for the android_total_ratings is : {}".format(lm.predict([4239138,230601,952604,514286,])))
print
print 

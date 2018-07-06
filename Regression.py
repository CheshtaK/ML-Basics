import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

import datetime
import matplotlib.pyplot as plt
from matplotlib import style

import pickle

dataframe = quandl.get('WIKI/GOOGL', authtoken = 'p8N5SQDgNJy7FaC1A3tY')
##print(dataframe.head())

dataframe = dataframe[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

dataframe['HL_PCT'] = (dataframe['Adj. High'] - dataframe['Adj. Low']) / dataframe['Adj. Low'] * 100.0

dataframe['PCT_change'] = (dataframe['Adj. Close'] - dataframe['Adj. Open']) / dataframe['Adj. Open'] * 100.0

dataframe = dataframe[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
##print(dataframe.head())

forecast_col = 'Adj. Close'
dataframe.fillna(value = -99999, inplace = True)
forecast_out = int(math.ceil(0.01 * len(dataframe)))

dataframe['label'] = dataframe[forecast_col].shift(-forecast_out)

X = np.array(dataframe.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

dataframe.dropna(inplace = True)

y = np.array(dataframe['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

pickle_in = open('linearregression.pickle', 'rb')
classifier = pickle.load(pickle_in)

##LinearRegression
##classifier = LinearRegression(n_jobs = -1)
##classifier.fit(X_train, y_train)
confidence = classifier.score(X_test, y_test)
##print('LinearRegression', confidence)

###svm.SVR
##for k in ['linear', 'poly', 'rbf', 'sigmoid']:
##    classifier = svm.SVR(kernel = k)
##    classifier.fit(X_train, y_train)
##    confidence = classifier.score(X_test, y_test)
##    print(k, confidence)


##with open('linearregression.pickle', 'wb') as f:
##    pickle.dump(classifier, f)

forecast_set = classifier.predict(X_lately)
print(forecast_set, confidence, forecast_out)

style.use('ggplot')
dataframe['Forecast'] = np.nan

last_date = dataframe.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    dataframe.loc[next_date] = [np.nan for _ in range(len(dataframe.columns)-1)]+[i]

dataframe['Adj. Close'].plot()
dataframe['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

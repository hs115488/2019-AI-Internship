#Author: Ambar Ruiz
#Contributors: Haresh Singh
#Github Username:
#   Ambar Ruiz: UNKNOWN
#   Haresh Singh: hs115488


import os
import numpy
import pandas
from datetime import datetime
from matplotlib import pyplot as plot
from matplotlib import pylab as plt
from matplotlib.pylab import rcParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import confusion_matrix, accuracy_score, classification_report

class RandomForest:
    def __init__(self, file):
        self.file = file


    def random_forest_algo(self):
        #Formatting data
        data = pandas.read_csv(self.file)
        data['Date'] = pandas.to_datetime(data['Date'], infer_datetime_format = True)
        date = data.set_index(['Date'])
        
        #Features Construction
        data['Open-Close'] = (data.Open - data.Close)/data.Open
        data['High-Low'] = (data.High - data.Low)/data.Low
        data['percent_change'] = data['Adj Close'].pct_change()
        data['std_2'] = data['percent_change'].rolling(2).std()
        data['ret_2'] = data['percent_change'].rolling(2).mean()
        data.dropna(inplace=True)
       
       # X is the input variable
        X = data[['Open-Close', 'High-Low', 'std_2', 'ret_2']]

        # Y is the target or output variable
        y = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, -1)

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        # Fitting Random Forest Classification to the Training set
        classifier = RandomForestClassifier(n_estimators = 900, criterion = 'gini', random_state = 0)
        classifier.fit(X_train, y_train)

        # Predicting Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

            #print('Correct Prediction (%): ', accuracy_score(y_test, model.predict(X_test), normalize=True)*100.0)
        
        #Running code to view classification report metrics
        report = classification_report(y_test, model.predict(X_test))
            #print(report)
        data['strategy_returns'] = data.percent_change.shift(-1) * model.predict(X)
        
        #Plotting Classifier
        %matplotlib inline
        data.strategy_returns[split:].hist()
        plt.xlabel('Strategy returns (%)')
        plt.show()

        #Output variable: If tomorrow’s close price is greater than today's close price then 
        #the output variable is set to 1 and otherwise set to -1. 1 indicates to buy the stock
        #and -1 indicates to sell the stock
        

        #Daily Returns Histogram

    def strategy_returns(self):
         
        %matplotlib inline
        rcParams['figure.figsize'] = 10,4

        (data.strategy_returns[split:]+1).cumprod().plot()
        plt.ylabel('Strategy returns (%)')
        plt.show()

        ##The output displays the strategy returns and daily returns according to the code for 
        #the Random Forest Classifier.
       


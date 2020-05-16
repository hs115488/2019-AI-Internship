#Author: Haresh Singh
#Github Username: hs115488


import pandas 
import numpy
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SupportVectorRegression:
    def __init__(self, file):
        self.file = file

    def support_vector(self):
        self.file = pandas.read_csv(self.file)
        dates = self.file['Date']
        price = self.file['Close']
        
        #Since numpy does not accept datetime format
        #for loop was created to count days
        #Integer value accepted

        days = []
        for i in range(0, len(dates)):
            days.append(i)

        #Fitting the model
        linear = SVR(kernel = 'linear', gamma = 'auto')
        X_train, X_test, y_train, y_test = train_test_split(days, price)
        linear.fit(X_train, y_train)
        predict = linear.predict(days)
        
        plt.scatter(days, price)
        plt.plot(days, price,color = "red", lw = "3")

        #Calculating MEAN SQUARED ERROR

        mse = mean_squared_error(price, predict)
        print("MSE: ", mse)
        print("Accuracy Score: ", linear.score(days, predict))

        
        

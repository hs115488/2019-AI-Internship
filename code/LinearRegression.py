#Author: Haresh Singh
#Github Username: hs115488

import pandas
import numpy
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

class Regression:
    def __init__(self, file):
        self.file = file

    def regression_models(self):
        self.file = pandas.read_csv(self.file)
        dates = self.file['Date']
        price = self.file['Close']
        
        #sklearn cannot accept datetime format
        #for loop created as counter to convert to integer value

        dates_count = []
        for i in range(0, len(date)):
            dates.append(i)
        dates_count = numpy.reshape(dates, (-1, 1))

        #Polynomial Regression
        #To understand volatility in simpler format

        poly = PolynomialFeatures(degrees = 2)
        X_poly = poly.fit_transform()
        poly.fit(X_poly, price)

        #Linear Regression
        
        lin2 = linear_model.LinearRegression()
        lin2.fit(X_poly, price)

        regr = linear_model.LinearRegression()
        regr.fit(dates_count, price)

        p = regr.predict(dates_count, price)
        q = lin2.predict(poly.fit_transform(dates_count)
        

        #Plot

        plt.scatter(dates_count, price)
        plt.plot(dates_count, p, color = "red")
        plt.plot(dates_count, q , color = "green")
        
        #Print Accuracy Score
        print(regr.score(dates_count, price))







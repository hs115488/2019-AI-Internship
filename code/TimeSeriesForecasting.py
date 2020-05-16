import pandas
import numpy as np
from matplotlib import pyplot as plt


class Forecast():

    def __init__(self, file):
        # Class needs a csv file as the dependency of this script
        # is the pandas library.
         
        self.file = file
     
    def plot_graph(self):
        #Closing prices was used to create a synposis to understand directionality
        self.file = pandas.read_csv(self.file)
        closing_prices = self.file['Close']
        closing_dates = self.file['Date']

        plt.scatter(closing_dates, closing_prices)
        #plt.plot(closing_dates, closing_prices)
        plt.show()

    def time_series_forecast(self):
        #Dependencies: fbprophet
        #To install, please 'python -m pip install fbprophet' 
        
        model = fbprophet.Prophet()
        date = file['ds']
        price = file['y']
        
        future = model.make_future_dataframe(periods = 365)
        forecast = model.predict(future)

        model.plot(forecast)
        model.plot_components(forecast)




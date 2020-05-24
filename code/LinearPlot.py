import pandas
from matplotlib import pyplot as plt

class Plot(object):
    def __init__(self, file):
        self.file = file

    def plot(self):
        self.file = pandas.read_csv(self.file)
        date = self.file["Date"]
        close = self.file["Close"]
        
        plt.plot(date, close)
        plt.show()

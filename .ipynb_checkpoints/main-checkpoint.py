# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# % matplotlib inline

from datetime import datetime


import re
import string
import nltk
#Download English Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_ENG = stopwords.words('english')

import pandas as pd

class APP():
    def __init__(self, path, headnum=1, separator=','):
        self.path = path
        self.headnum = headnum
        self.separator = separator
        self.df = self.load_data()

    def load_data(self):
        # Read data from CSV file
        df = pd.read_csv(self.path, sep=self.separator, header=self.headnum)
        return df

if __name__ == "__main__":
    # Instantiate the DataReader class
    reader = DataReader('/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
    # View first few columns of data
    print(reader.df.head())

# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# % matplotlib inline
from preprocessing import *
from datetime import datetime
from modeling import *


import re
import string
import nltk
#Download English Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_ENG = stopwords.words('english')

import pandas as pd

keepcolumns = [ 'price','minimum_nights', 'number_of_reviews', 'neighbourhood_group',
       'room_type','calculated_host_listings_count', 'reviews_per_month', 'neighbourhood',
       'availability_365']

num_features = ['price','minimum_nights','number_of_reviews','reviews_per_month',
                'calculated_host_listings_count','availability_365']

class APP():
    def __init__(self, path, keepcolumns ,numfeatures,headnum=1, separator=','):
        self.path = path
        self.headnum = headnum
        self.separator = separator
        self.keepcolumns = keepcolumns
        self.numfeatures = numfeatures
        self.df = self.load_data()
        self.X_train,self.y_train,self.X_test,self.y_test = self.preprocess_data(self.df)
        self.train_models()
    def load_data(self):
        # Read data from CSV file
        df = pd.read_csv(self.path, sep=self.separator, header=self.headnum)
        return df
    
        # @ANNABELLE please check this
        # if __name__ == "__main__":
        #     # Instantiate the DataReader class
        #     reader = DataReader('/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
        #     # View first few columns of data
        #     print(reader.df.head())

    def preprocess_data(self):
        df = fill_missing_values(self.df)
        df  = scale_data(df,num_features)
        df = one_hot_data(df,['neighbourhood_group','room_type'])
        df = encode_data(df,['neighbourhood'])
        X_train,y_train,X_test,y_test = split_data(df,self.keepcolumns)
        return X_train,y_train,X_test,y_test

    def train_models(self):
        train_linear_reg(self.X_train,self.y_train,self.X_test,self.y_test)

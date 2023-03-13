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


def preprocess_data(df, num_features):
    df = fill_missing_values(df)
    df  = scale_data(df,num_features)
    df = one_hot_data(df,['neighbourhood_group','room_type'])
    df = encode_data(df,['neighbourhood'])
    X_train,y_train,X_test,y_test = split_data(df,keepcolumns)
    return X_train,y_train,X_test,y_test

def train_models():
    train_linear_reg(X_train, y_train, X_test, y_test)

"""
This is the main module to run the script and functions defined 
"""
if __name__ == "__main__":
    # read the csv file as a dataframe
    path = 'new-york-city-airbnb-open-data/AB_NYC_2019.csv'
    df = pd.read_csv(path, sep=',')
    X_train,y_train,X_test,y_test = preprocess_data(df, num_features)
    train_models()
    '''
    preprocessing functions ()
    train()
    etc.
    '''

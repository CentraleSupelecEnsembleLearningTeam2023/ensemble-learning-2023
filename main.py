# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# % matplotlib inline
from preprocessing import *
from datetime import datetime
from modeling import *
from feature_engineering import *


import re
import string
import nltk
#Download English Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_ENG = stopwords.words('english')

import pandas as pd

keepcolumns = ['price','minimum_nights', 'number_of_reviews', 'neighbourhood_group',
       'room_type','calculated_host_listings_count', 'reviews_per_month', 'neighbourhood',
       'availability_365','last_review_recency']

num_features = ['price','minimum_nights','number_of_reviews','reviews_per_month',
                'calculated_host_listings_count','availability_365','last_review_recency']


def preprocess_data(df, num_features,keepcolumns):
    df = fill_missing_values(df) #Add missing values

    df  = scale_data(df,num_features) #scale numerical features
    df = one_hot_data(df,['neighbourhood_group','room_type']) #one hot encode small category data
    df,encoded_columns = encode_data(df,['neighbourhood']) #encode large category data

    for column in ['neighbourhood_group','room_type']:#modify keep columns to add one hot encode changes
        keepcolumns.remove(column)
    keepcolumns = keepcolumns + encoded_columns

    X_train,y_train,X_test,y_test = split_data(df,keepcolumns) #split data into train and test set

    return X_train,y_train,X_test,y_test

def train_models():
    train_linear_reg(X_train, y_train, X_test, y_test, cross_val = False)
    train_decision_tree(X_train,y_train,X_test,y_test,max_depth= None,min_samples_split = 2,cross_val = False,grid_search = False)
    train_random_forest(X_train,y_train,X_test,y_test,estimators = 100,max_depth= None,min_samples_split = 2,cross_val = False,grid_search = False)

"""
This is the main module to run the script and functions defined 
"""
if __name__ == "__main__":
    # read the csv file as a dataframe
    path = 'new-york-city-airbnb-open-data/AB_NYC_2019.csv'
    df = pd.read_csv(path, sep=',')
    X_train,y_train,X_test,y_test = preprocess_data(df, num_features,keepcolumns)
    train_models()
    '''
    preprocessing functions ()
    feature engineering()
    train()
    etc.
    '''

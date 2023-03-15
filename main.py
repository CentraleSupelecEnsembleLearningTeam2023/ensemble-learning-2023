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


def add_engineered_features(df):
    '''
    Function to add features engineered in aggregrate manner
    '''
    # Read in the new york subway stations location data data
    # Source: https://data.ny.gov/en/Transportation/NYC-Transit-Subway-Station-Map/6xm2-7ffy
    stations = pd.read_csv('new-york-city-airbnb-open-data/NYC_Transit_Subway_Station_Location.csv')
    #from tqdm import tqdm, tqdm_pandas
    #tqdm.pandas()

    df_transformed = df.copy()
    df_transformed['last_review_recency'] = recency(df_transformed)
    df_transformed['count_words_in_listing'] = count_words_in_listing(df_transformed)
    df_transformed['num_of_stations_nearby'] = df_transformed.apply(lambda row: numstation(row['latitude'], row['longitude'], stations), axis = 1)
    # Use the following line if you want to track the progress
    #df_transformed['num_of_stations_nearby'] = df_transformed.progress_apply(lambda row: numstation(row['latitude'], row['longitude'], stations), axis = 1)

    return df_transformed

def preprocess_data(df,columns_to_keep,cat_features,num_features,plot_dist = False,transform = True,encode = False,
                    one_hot_features = None, test_size  = 0.2):
  '''
  Function to preprocess data for modelling
  Parameters:
  -----------------------
  df - dataframe to preprocess
  columns_to_keep - columns to be used for modelling
  cat_features - categorical features (must be in columns_to_keep)
  num_features  - nuemerical features (must be in columns_to_keep)
  plot_dist - Boolean if want to view distribution of numerical data after transformation
  tranform - Boolean if numerical data to be log transformed
  encode - Boolean if categorical data to be encoded
  one_hot_features - Boolean if enode is True, features to be 1hot encoded. rest of categorical features
                    will be encoded into numerical data
  test_size - size of test set

  Returns:
  -------------------------
  X_train - training features
  X_test - test features
  y_train - training target
  y_test - test target
  '''
  df_transformed = df.copy()
  kept_columns = columns_to_keep.copy()
  df_transformed = fill_missing_values(df_transformed) # missing value imputation
  if transform == True:
    df_transformed = log_data(df_transformed,num_features) #log tranforms numerical features
  
  df_transformed = scale_data(df_transformed,num_features) #scales numerical data
  
  if plot_dist ==True:
    plot_density_per_num_column(df_transformed,num_features) #plot dist numerical columns

  if encode == True:
    df_transformed, encoded_columns = one_hot_data(df_transformed,one_hot_features) #1hot encode selected cat features

    for column in one_hot_features:
      kept_columns.remove(column)

    kept_columns = kept_columns + encoded_columns

    #numerically encodes rest of categorical features
    df_transformed = encode_data(df_transformed,[column for column in cat_features if column not in one_hot_features])
  
  #split data into train and test set
  X_train,X_test,y_train,y_test = split_data(df_transformed, columns_to_keep = kept_columns,test_size = test_size)
  
  return X_train,y_train, X_test, y_test

"""
considering multiple models, and some have different preoricessing, e.g. catboost,
we should abandon this function and move the functions to the main module
"""
def train_models(X_train, y_train, X_test, y_test):
    #train_linear_reg(X_train, y_train, X_test, y_test, cross_val = False)
    #train_decision_tree(X_train,y_train,X_test,y_test,max_depth= None,min_samples_split = 2,cross_val = False,grid_search = False,transform = False)
    #train_random_forest(X_train,y_train,X_test,y_test,estimators = 100,max_depth= None,min_samples_split = 2,cross_val = False,grid_search = False, transform = False)
    train_ensemble_models(X_train,y_train)
    train_lgbm(X_train,y_train,X_test,y_test)

"""
This is the main module to run the script and functions defined 
"""
if __name__ == "__main__":
    # read the csv file as a dataframe
    path = 'new-york-city-airbnb-open-data/AB_NYC_2019.csv'
    df = pd.read_csv(path, sep=',')

    #add engineered features
    df = add_engineered_features(df)

    #columns to keep and define numerical and categorical
    keepcolumns = ['price','minimum_nights', 'number_of_reviews', 'neighbourhood_group',
       'room_type','calculated_host_listings_count', 'reviews_per_month', 'neighbourhood',
       'availability_365','last_review_recency','count_words_in_listing','num_of_stations_nearby']

    num_features = ['price','minimum_nights','number_of_reviews','reviews_per_month',
                'calculated_host_listings_count','availability_365','last_review_recency',
                'count_words_in_listing','num_of_stations_nearby']
    
    cat_features = ['neighbourhood_group','neighbourhood','room_type']
    
    #pre process data
    X_train_encoded,y_train_encoded, X_test_encoded, y_test_encoded = preprocess_data(df, columns_to_keep=keepcolumns, cat_features=cat_features,
                                                   num_features=num_features, plot_dist=False, encode=True,
                                                   one_hot_features=['neighbourhood_group','room_type'],
                                                   test_size=0.2)
    X_train, y_train, X_test, y_test = preprocess_data(df, columns_to_keep=keepcolumns, cat_features=cat_features,
                                                   num_features=num_features, plot_dist=False, encode=True,
                                                   one_hot_features=['neighbourhood_group','room_type'],
                                                   test_size=0.2)

    #train models
    train_ensemble_models(X_train_encoded, y_train_encoded)

    train_lgbm(X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded, estimators=200, lr=0.07, n_jobs=-1, rs=42,
               transform=False)
    train_xgb(X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded, estimators=200, lr=0.07, rs=42, transform=False,
              device='cpu', max_depth=6)
    train_catboost(X_train, y_train, X_test, y_test, estimators=3000, lr=1 / 10, max_depth=6,
                   l2=5, eval_metric="R2", one_hot_max_size=1000, od_type="Iter", od_wait=0,
                   transform=False, verbose=False, data_in_leaf=1)
    '''
    preprocessing functions ()
    feature engineering()
    train()
    etc.
    '''

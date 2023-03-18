#Functions to be used for preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #For Data normalization
from sklearn.model_selection import train_test_split #To split train test data
from sklearn import preprocessing # label encode neighborhood
import matplotlib as plt


# Define the function to fill missing values
def fill_missing_values(df,columns):

  ''' Fills missing values in data frame.
  Null values in last_review & reviews_per_month coincide with each other indicating 
  that the listing never had a review
  This implies it is logical to fill reviews_per_month with 0 and last_review with the minimum review date'''

  if 'reviews_per_month' in columns:
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0) #fill null reviews per months by 0
  if 'last_review' in columns:
    df['last_review'] = pd.to_datetime(df['last_review']) #convert to datetime
    df['last_review'] = df['last_review'].fillna(min(df.last_review)) #fill in with last review with the most recent date in dataset
  if 'name' in columns:
    df['name'] = df['name'].fillna('') #fill missing listing name within no name

  return df

# Define the function to apply boxcox transformation
def transform_boxCox(df,num_features):
  '''Apply BoxCox transfromation to columns to improve data distribution
  of numerical variables
  '''
  ## Box cox transformation (not removing outliers)
  # apply Box-Cox transformation to the selected columns
  from scipy.stats import boxcox
  epsilon = 0.001  # to keep strictly positive bound for box cox transformation
  df_transformed = df.copy()
  for col in num_features:
    df_transformed[col],_ = boxcox(df_transformed[col]+epsilon)
  return df_transformed

#Define the function to log transform data
def log_data(df,num_features):
  '''
  returns natural log of numerical data
  '''
  df_transformed = df.copy()
  for col in num_features:
    if col == 'price':
        df_transformed[col] = np.log(df_transformed[col])
    else:
        df_transformed[col] = np.log(df_transformed[col]+1)

  return df_transformed

# Define the function to scale the input data
def scale_data(df,num_features):
    '''Scales the data to a specific using Min Max Scalar method so that
    all numerical data in the same scale and feature importance not biased
    by large data
    '''
    df_transformed = df.copy()
    # Create scaler object
    scaler = MinMaxScaler()

    # Scale data
    scaler.fit(df[num_features])
    scaler = scaler.transform(df[num_features])
    df_transformed[num_features] = scaler
    return df_transformed

# Define the function to split the training data into training set and test set
def split_data(df,columns_to_keep,test_size = 0.2):
    '''Splits the dataset into a training and test set
    and returns the features and targets for the training
    and test set
    '''

    df_to_keep = df[columns_to_keep]

    cols = df_to_keep.columns.tolist()
    cols.remove('price')
    X = df_to_keep[cols]
    y = df_to_keep[['price']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state = 123)

    return X_train,X_test,y_train,y_test

# Define the function to conduct the one-hot encoding for the categorical features
def one_hot_data(df, cat_features):
  '''one hot encodes categorical selected categorical features
  Returns:
  ------------------
  df_encoded - dataframe with one_hot encoded columns
  encoded_columns - list of columns names of one hot encoded columns
  '''
  unique_categories = 0
  for column in cat_features:
    unique_categories += df[column].nunique()

  df_encoded = pd.get_dummies(df, columns=cat_features)

  encoded_columns = list(df_encoded.columns[len(df.columns)+ unique_categories - 1::-1][:unique_categories])

  return df_encoded, encoded_columns

# Define the function to conduct the label encoding for the categorical features
def encode_data(df,cat_features):
    '''label encoding encodes categorical selected categorical features
        Returns: (eg:0,1,2...)
        ------------------
        df_encoded - dataframe with label encoded columns
    '''

    le = preprocessing.LabelEncoder()
    df_encoded = df.copy()
    for column in cat_features:
        le = preprocessing.LabelEncoder()
        le.fit(df_encoded[column])
        df_encoded[column] = le.transform(df_encoded[column])

    return df_encoded

# Define the function that plots the distribution of numerical features
def plot_density_per_num_column(df,numerical_features):
  '''Function to plot the distribution of numerical features
  '''
  cols = 2
  rows = int(np.ceil(len(num_features)/cols))

  fig = plt.figure(figsize=(6*cols,5*rows))
  for i, fea_name in enumerate(numerical_features):
    ax = fig.add_subplot(rows,cols,i+1)
    df.loc[:,fea_name].hist(color='green',alpha=0.5,rwidth=0.8, density=True,
                            grid=False,ax=ax, bins=20)
    plt.axvline(df.loc[:,fea_name].mean(),
                color='r',linestyle=':',label = 'mean')
    plt.axvline(df.loc[:,fea_name].median(), color='b', linestyle='-.',
                label = 'median')
    plt.legend()
    #df.loc[:,fea_name].plot(kind='density', color='green')
    ax.set_title("Histogram of " + fea_name)

  plt.show()
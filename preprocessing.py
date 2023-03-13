#Functions to be used for preprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #For Data normalization
from sklearn.model_selection import train_test_split #To split train test data
from sklearn import preprocessing # label encode neighborhood

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
    Returns: (eg: 0,1)
    ------------------
    df_encoded - dataframe with one_hot encoded columns
    '''
    df_encoded = pd.get_dummies(df, columns=cat_features)
   
    return df_encoded

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
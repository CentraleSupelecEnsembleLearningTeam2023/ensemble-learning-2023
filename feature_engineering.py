#Functions of engineered features
import pandas as pd
import numpy as np

def recency(df):
    """
    Create new feature which looks at the recency of the of reviews by taking the difference
  between the latest review date of the listing and the latest review date in the entire
  dataset.
    :param df: dataframe
    :return: recency days in pandas series
    """
    df_last_review = pd.DataFrame(pd.to_datetime(df['last_review']), columns=['last_review'])
    df_last_review['last_review'] = df_last_review['last_review'].fillna(min(df_last_review['last_review']))
    recency = df_last_review['last_review'].max() - df_last_review['last_review']

    return recency.astype('timedelta64[D]')

def count_words_in_listing(df):
  ''' 
  This function will take the string of name description of
  each listing and count the number of words of that listing.
  '''
  df_transformed=df.copy()

  return [len(str(name).split()) for name in df_transformed["name"]]

def distance(lat1, lon1, lat2, lon2):
    '''
    Calculates the distance in meters between two coordinates on Earth, given their latitudes and longitudes in degrees.

    Args:
    - lat1: float - latitude of the first point in degrees
    - lon1: float - longitude of the first point in degrees
    - lat2: float - latitude of the second point in degrees
    - lon2: float - longitude of the second point in degrees
    
    Returns:
    - distance: float - the distance between the two points in meters
    '''
    R = 6371
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)*2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)*2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c * 1000
    
    return distance

def numstation(la, lo, df2):
    '''
    Counts the number of stations within a radius of 1 km, given a latitude and longitude representing a location, and a dataframe of public transport stations.

    Args:
    - la: float -  the latitude of the location
    - lo: float -  the longitude of the location
    - df2: pandas DataFrame - A dataframe of public transport stations with columns 'Station Latitude' and 'Station Longitude'

    Returns:
    - count: integer - the number of public transport stations within 1km of the location
    '''
    
    lat2 = df2['Station Latitude'].values
    lon2 = df2['Station Longitude'].values
  
    distances = distance(np.radians(la), np.radians(lo), np.radians(lat2), np.radians(lon2))

    # Count the number of public transport stations within 1000 meters from the house
    count = np.sum(distances <= 1000)
    
    return count

def min_landmark_distance(df_lat, df_long, df_land):

  '''Create features that calculates the minimum distance between a listing
  and one of the Top 13 landmarks. The function will calculate the distance of a listing
  from from all the landmarks and output the distance to the closest landmark. Uses 
  widely used methodology called Haversine to calculate the distance between the coordinates
  '''

  from math import sin, cos, sqrt, atan2, radians
  
  df_lat = df_lat.apply(radians) #convert latitude of all listings to radians
  df_long = df_long.apply(radians) #convert longitude of all listings to radians
  R = 6372 #radius of Earth
  i = 0
  
  #iterate over landmarks
  for landmark_lat,landmark_long in df_land[['latitude','longitude']].values:
    landmark_lat = radians(landmark_lat) #convert latitude of landmark to radians
    landmark_long = radians(landmark_long) #convert longitude of landmark to radians
  
    dlon = landmark_long - df_long #difference between longitude of listings and landmark
    dlat = landmark_lat - df_lat #difference between latitude of listings and landmark
    
    #Usese the Haversine formula for calculating distance between cpprdinates
    a = ((dlat / 2).apply(sin))**2 + df_lat.apply(cos) * cos(landmark_lat) * ((dlon / 2).apply(sin))**2
    c = 2* a.apply(lambda x: atan2(sqrt(x),sqrt(1-x)))

    distance_haversine_formula = R * c
    #creates a tuple with harvesine distance between all listings and first landmark
    if i == 0:
      distance_from_landmarks = tuple(distance_haversine_formula)
    #zips distances from second landmarks
    elif i == 1:
      distance_from_landmarks = zip(distance_from_landmarks,tuple(distance_haversine_formula))
    #zips distances from rest of landmarks
    else:
      distance_from_landmarks = list(map(lambda x: list(x[0]) + [x[1]],list(zip(distance_from_landmarks,distance_haversine_formula))))

    i +=1

  #takes the distance to the closest landmark for each respective listing
  min_distance_from_landmarks = [min(x) for x in distance_from_landmarks]
  
  return min_distance_from_landmarks

def clean_string(line,stopwords): 

  '''Function to clean text in "name" column to remove common stop words and remove punctuation 
  and other irrelevant information
  '''
  import re

  line = re.sub(r'[^\w\s]',"",line) #removes punctuation
  line = [word.lower() for word in line.split(" ") if (len(word)>0) & (word.lower() not in stopwords)] #Each lines is then split into words
  #by parsing blank space. any word that is a stop word is remove and then lower case for better gouping.

  return ' '.join(line) # the output is list of words due to the parsing op

def tokenize_text(df_train, stopwords, df_test = 0,test = False,min_df = 2,text_processed = True):

  '''Create Features of from "name" text that are token counts of the collection
  of text in the the entire dataframe using a CountVectorizer. The function will
  return a pandas dataframe of counts of all collection of tokens and their counts
  per listing. If a test set is specified, the process will also be made for a test 
  set.
  '''
  from sklearn.feature_extraction.text import CountVectorizer
  count_vect = CountVectorizer(min_df=min_df)

  #clean text if not yet processed
  if text_processed == False:
    df_train['name'] = [clean_string(text,stopwords) for text in list(df_train['name'])]

  #fit data to training set
  X_train_counts = count_vect.fit_transform(df_train['name'])

  #create list of column names of tokens
  token_columns = []
  for column in count_vect.get_feature_names_out():
    token_columns.append("token_"+column)

  #covert count vector into a Pandas DataFrame (to be able to merge with rest of features)
  count_vec_df_train = pd.DataFrame(X_train_counts.todense(), columns=token_columns,index = df_train.index)

  if test == True:

    if text_processed == False:
      df_test['name'] = [clean_string(text,stopwords) for text in list(df_test['name'])]

    X_test_counts = count_vect.transform(df_test['name'])
    print("Train shape:", X_train_counts.shape, "Test shape:", X_test_counts.shape)
    count_vec_df_test = pd.DataFrame(X_test_counts.todense(), columns=token_columns,index = df_test.index)
    return  count_vec_df_train, count_vec_df_test

  else:
    print("Train shape:", X_train_counts.shape)
    return count_vec_df_train
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
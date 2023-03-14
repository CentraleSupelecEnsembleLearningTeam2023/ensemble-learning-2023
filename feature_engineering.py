#Functions of engineered features
import pandas as pd

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


def add_engineered_features(df):
    '''
  Function to add features engineered in aggregrate manner
  '''
    df_transformed = df.copy()
    df_transformed['last_review_recency'] = recency(df_transformed)
    df_transformed['count_words_in_listing'] = count_words_in_listing(df_transformed)
    return df_transformed
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
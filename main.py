# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# % matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
from datetime import datetime


import re
import string
import nltk
#Download English Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_ENG = stopwords.words('english')

class App():
    def __init__(self):
        path = '/new-york-city-airbnb-open-data/AB_NYC_2019.csv'
        headnum = 1  # Change it according to the real case
        separater = ','  # Sometimes it's ';'
        # CSV file
        self.df = pd.read_csv(path, sep=separat
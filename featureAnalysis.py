"""
Justin Clark
CSYS 300
Final Project
featureAnalysis.py

Exploratory Data Analysis
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import os
import pandas as pd
import lyricsgenius
import re
import nltk
import numpy as np
from nltk.sentiment import SentimentAnalyzer
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 

def calculateTotalFreq(df):
    """

    
    """
    totalFrq = np.sum([Counter(df.iloc[i,-1]) for i in range(df.shape[0]) if df.iloc[i,-1] != None ])
    #totalFrq.most_common(10)
    sortedTotalFreq = sorted(totalFrq.items(), key=lambda pair: pair[1], reverse=True)
    return sortedTotalFreq

def calulateStrongestFeatureCorrelations(df):
    c= df.corr().abs()
    s = c.unstack()
    so = s.sort_values(kind = 'quicksort')
    print(so.tolist()[-20])

merged_df = pd.read_csv("final_merged_df.csv")
#merged_df = pd.read_csv(merged_df)
df_above_50 = merged_df[merged_df.popularity > 50]

calulateStrongestFeatureCorrelations(merged_df)
calulateStrongestFeatureCorrelations(df_above_50)

sns.distplot(df_above_50['popularity'])
plt.title("Loudness for Sounds above 50 Popularity")

artist_grouped_df = df_above_50.groupby(['artist']).mean()


list_artists = artist_grouped_df.index.values.tolist()
plt.scatter(artist_grouped_df['valence'],artist_grouped_df['popularity'])
for i,text in enumerate(list_artists):
    plt.annotate(text,(artist_grouped_df['valence'][i],artist_grouped_df['popularity'][i]))
    
sortedTotalFreq = calculateTotalFreq(merged_df)

#artis = merged_df.groupby(['artist']).sum()
# -*- coding: utf-8 -*-
"""
Justin Clark
CSYS 300
Final Project
mainScitpy.py

Data Proprocessing...
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
from nltk.stem import PorterStemmer
from num2words import num2words
from langdetect import detect 


#from spotify_api import tokenizeLyrics,sentimentAnalysis
stop_words = stopwords.words()
ps = PorterStemmer()
def preProcessData(genius,artist_name,track_name,stop_words):
    symbols = "!,\"#$%&()*+-./:;<=>?@[\]^_`{|}~"#\n"
    lyrics_string = genius.search_song(track_name,artist_name).lyrics
    lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics_string)
    lyrics = lyrics.lower()
    #return lyrics
    words = lyrics.split (' ')
    newtst = ''
    for word in words:
        if word not in stop_words:
            newtst = newtst+ " "+  word
    #return newtst
    for i in symbols:
        newtst = newtst.replace(i,'')
    newtst.replace("'","")
    #return newtst
    lines = newtst.split('\n')
    #return lines
    words = newtst.replace('\n',' ').split(' ')
    #return words
    #wordz = newtst.split(" ")
    #return wordz
    #final_lyrics = ''
    for word in words:
        if len(word) <= 1:
            words.remove(word)
    for line in lines:
        if len(line) <= 1:
            lines.remove(line)
    return lines,words
    #return final_lyrics
#    line_split = final_lyrics.split('\n')
    #return line_split
    #words = word_tokenize(line_split)
    #stem_words = [ps.stem(w) for w in words]
    #r#eturn words,stem_words
def tokenizeLyrics(genius,artist_name,track_name):
    """
    Tokenize Lyrics of song
    
    Arguments:
        genius:
        artist_name:
        track_name:
    Return:
        line_split_lyrics:
        word_frequency
    """
    #Initialize tokenizer
    tknzr = nltk.TweetTokenizer()
    lyrics_string = genius.search_song(track_name,artist_name).lyrics
    lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics_string).lower()
    linesplit_lyrics = lyrics.split('\n')
    text_tokens = tknzr.tokenize(lyrics)
    #text_tokens = word_tokenize(lyrics)
    print(text_tokens)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    #filtered_sentence = (" ").join(tokens_without_sw)
    print(tokens_without_sw)
    word_frequency = Counter(tokens_without_sw)
    return linesplit_lyrics,word_frequency
def sentimentAnalysis(line_split_lyrics,artist,song):
    """"
    Analysis sentiment of single song, calculting
    the compound polarity score of each line.
    
    Arguments:
        line_split_lyrics:
        artist:
        song:
    Return:
    """
    comp_list = []
    numberOfLines = 0
    numberPosLines = 0
    numberNeuLines = 0
    numberNegLines = 0
    sid = SentimentIntensityAnalyzer()
    while '' in line_split_lyrics:
        line_split_lyrics.remove('')
    for line in line_split_lyrics:
        numberOfLines += 1
        ss = sid.polarity_scores(line)
        if ss > 0:
            numberPosLines += 1
        elif ss == 0:
            numberNeuLines +=1
        elif ss < 0:
            numberNegLines += 1 
        comp_list.append(ss['compound'])
        #pos_list.append(ss[])
        #for k in sorted(ss):
         #print('{0}: {1}, '.format(k, ss[k]), end='')
         #print()
         #
#    plt.plot(comp_list)
#    plt.xlabel('Line Number')
#    plt.ylabel('Compound Polarity Score')
#    plt.title('{} by {}'.format(artist,song))
#    plt.show()
    avgSentiment = np.average(comp_list)
    proportionPositive = numberPosLines/numberOfLines
    proportionNeutral = numberNeuLines/numberOfLines
    proportionNegative = numberNegLines/numberOfLines
    return avgSentiment,proportionPositive,proportionNeutral,proportionNegative

def calculateTotalFreq(df):
    """

    
    """
    totalFrq = np.sum([df.iloc[i,-1] for i in range(df.shape[0]) if df.iloc[i,-1] != None ])
    #totalFrq.most_common(10)
    sortedTotalFreq = sorted(totalFrq.items(), key=lambda pair: pair[1], reverse=True)
    return sortedTotalFreq

def addYearCol(df,year):
    """
    
    
    """
    year_list = []
    for i in range(df.shape[0]):
        year_list.append(year)
    df['Year'] = year_list
        

################################
### SPOTIPY ####################
os.environ['SPOTIPY_CLIENT_ID'] = '1e19cf87b9524b8bacd09a98c4eef4c4'
os.environ['SPOTIPY_CLIENT_SECRET'] = 'facdf5ffa42e40ae9e335bebaeea0609'
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888/callback/'
auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)
genius_client_id = '9g5ycExMAL9fWr_r0Y9Lyf7aoZDIYei8FN4ENPf1LZe6RjFX4sBT3xbULYB9qa7x'
genius_client_secret = 'oIQt0GoOWNP9wcVM2yn2tqTbhYAcPp4YOdjNu4VSkB1coCoM0FG7_Nh6az9f0JTYGMdzeEo47OCbhu3y14L4hQ'
genius_client_access_token = 'vequbCKgDuQVe3UtDd-RD7dzuG2sMxZaNz_J8pELZk01SXTK2KvxDZK68E2NtJ9h'
genius = lyricsgenius.Genius(genius_client_access_token)
sentim_analyzer = SentimentAnalyzer()
####################################


df_2019 = pd.read_csv("rap_df_2019.csv")
addYearCol(df_2019,2019)
df_2018 = pd.read_csv("df_2018.csv")
addYearCol(df_2018,2018)
df_2017 = pd.read_csv("df_2017.csv")
addYearCol(df_2017,2017)
df_2016 = pd.read_csv("rap_df_2016.csv")
addYearCol(df_2016,2016)
df_2015 = pd.read_csv("rap_df_2015.csv")
addYearCol(df_2015,2015)
df_2014 = pd.read_csv("rap_df_2014.csv")
addYearCol(df_2014,2014)
df_2013 = pd.read_csv("rap_df_2013.csv")
addYearCol(df_2013,2013)
df_2012 = pd.read_csv("rap_df_2012.csv")
addYearCol(df_2012,2012)

frames = [df_2019,df_2018,df_2017,df_2016,df_2015,df_2014,df_2013,df_2012]
merged_df = pd.concat(frames)


num_unique_artists = len(set(merged_df['artist'].tolist()))


list_of_words = []
avgSentiment = []
for num_rows in range(merged_df.shape[0]):
    print('Percent Done: {}'.format((num_rows/merged_df.shape[0])*100))
    artist = merged_df.iloc[num_rows,0]
    track = merged_df.iloc[num_rows,2]
    try:
        lines,word_list = preProcessData(genius,artist,track,stop_words)
#        if detect(lines[0]) != 'en':
#            list_of_words.append(None)
#            avgSentiment.append(None)
#        else:
        list_of_words.append(word_list)
        #df_2019.replace(num_rows,-1) = word_freq
        #a,p,neu,n = sentimentAnalysis(lines,artist,track)
        #avgSentiment.append(a)
    except:
        print('Sorry, Song Lyrics not Found')
        list_of_words.append(None)
        #avgSentiment.append(None)
        #df_2019.loc[num_rows,'Word Frequency'] = None
merged_df['Words'] = list_of_words
#merged_df['Avg Sentiment'] = avgSentiment
merged_df.to_csv("final_merged_df.csv",index = False)

# -*- coding: utf-8 -*-
"""
Justin Clark
CSYS 300

"""

####################
### IMPORT ##
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import os
import pandas as pd
import lyricsgenius
import re
import nltk
from nltk.sentiment import SentimentAnalyzer
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
################################

def analyze_playlist(creator, playlist_id):
    
    # Create empty dataframe
    playlist_features_list = ["artist","album","track_name",  "track_id","popularity","danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    # Loop through every track in the playlist, extract features and append the features to the playlist df
    
    playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    for track in playlist:
        # Create empty dict
        playlist_features = {}
        # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]
        playlist_features["popularity"] = track["track"]["popularity"]
        # Get audio features
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[5:]:
            playlist_features[feature] = audio_features[feature]
        
        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
        
    return playlist_df

def createYearlyDataFrames(dictionary):
    """
    Create pandas dataframe for each year and save file to
    csv for later use
    Arguments:
        dictionary:
            key = String of Year
            value[0] = Creator ID
            value[1] = Playlist ID
    """
    for key in dictionary:
        year = key
        creator = dictionary[key][0]
        playlist_id = dictionary[key][1]
        df = analyze_playlist(creator,playlist_id)
        df.to_csv("df_{}.csv".format(year),index = False)
        

################################
### SPOTIPY ####################

os.environ['SPOTIPY_CLIENT_ID'] = '1e19cf87b9524b8bacd09a98c4eef4c4'
os.environ['SPOTIPY_CLIENT_SECRET'] = 'facdf5ffa42e40ae9e335bebaeea0609'
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888/callback/'
auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)
###############################
genius_client_id = '9g5ycExMAL9fWr_r0Y9Lyf7aoZDIYei8FN4ENPf1LZe6RjFX4sBT3xbULYB9qa7x'
genius_client_secret = 'oIQt0GoOWNP9wcVM2yn2tqTbhYAcPp4YOdjNu4VSkB1coCoM0FG7_Nh6az9f0JTYGMdzeEo47OCbhu3y14L4hQ'
genius_client_access_token = 'vequbCKgDuQVe3UtDd-RD7dzuG2sMxZaNz_J8pELZk01SXTK2KvxDZK68E2NtJ9h'
genius = lyricsgenius.Genius(genius_client_access_token)
sentim_analyzer = SentimentAnalyzer()

#df_2019 = analyze_playlist('AT MusicPedia','1YEm3mSbOeDnoftDjSkcYz')
dictionary_of_playlists = {
        "2019":['AT MusicPedia','1YEm3mSbOeDnoftDjSkcYz'],
        "2018":["DomSki",'6SWqTtjpGxmWW8eDwWQDuH'],
        "2017":["boardboymusicworldwide",'7LJTmZfNGAg8lsbiFVsNSx']
        }

createYearlyDataFrames(dictionary_of_playlists)

# Collect 1,000 Track IDs and associated name, artists name
# and popularity. score



#artist_name = []
#track_name = []
#popularity = []
#track_id = []
#for i in range(0,10000,50):
#    track_results = sp.search(q='year:2019', type='track', limit=50,offset=i)
#    for i, t in enumerate(track_results['tracks']['items']):
#        artist_name.append(t['artists'][0]['name'])
#        track_name.append(t['name'])
#        track_id.append(t['id'])
#        popularity.append(t['popularity'])
#        
#track_dataframe = pd.DataFrame({'artist_name' : artist_name, 'track_name' : track_name, 'track_id' : track_id, 'popularity' : popularity})
#print(track_dataframe.shape)
#track_dataframe.head()


############################
### GENIUS API



#artist = genius.search_artist('Juice WRLD', max_songs=3, sort="title",include_features = True)
#print(artist.songs)


#nltk.word_tokenize(lucid_lyrics)

### Sentiment Analysis
#for line in lucid_lyrics.replace('?','').replace(',','').split('\n'):



#nltk.word_tokenize(genius.search_song('Dior','Pop Smoke').lyrics)
#nltk.wordpunct_tokenize(genius.search_song('Dior','Pop Smoke').lyrics)
#Counter(nltk.wordpunct_tokenize(genius.search_song('Lucid Dreams','Juice WRLD').lyrics))
#Counter(nltk.wordpunct_tokenize(genius.search_song('Lucid Dreams','Juice WRLD').lyrics.lower()))


## SAVE DF TO CSV
#df_2019.to_csv("2019_rap_top100.csv", index = False)

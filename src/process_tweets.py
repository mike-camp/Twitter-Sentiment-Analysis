"""A collection of modules to scrape tweets from a mongo database
and then process the location, and the sentiment of the tweet.
"""
import re
import pickle
import utils
import pandas as pd

model = None

def find_state(tweet):
    """given a tweet finds the state it corresponds to
    and then returns tweet with state added in as a column
    """
    states = pd.read_csv('../data/states.csv')
    full_names = set(states.State)
    abbreviations = set(states.Abbreviation)
    location = tweet['user']['location']
    for state in full_names:
        if state.lower() in location.lower():
            return state
    parts_of_location = re.split(r'\[^a-zA-Z]+',location.lower())
    for state in abbreviations:
        if state in parts_of_location:
            return state

def find_sentiment(tweet):
    tweet_text = tweet['extended_tweet']['full_text']
    probability = model.predict_proba(tweet_text)

if __name__=='__main__':
    with open('../models/text_model/pkl','rb') as f:
        model = pickle.load(f)
    



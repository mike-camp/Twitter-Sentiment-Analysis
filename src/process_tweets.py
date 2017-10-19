"""A collection of modules to scrape tweets from a mongo database
and then process the location, and the sentiment of the tweet.
"""
import re
import pickle
import os
import pandas as pd
from src import model
from src.model import TweetPredictor
import datetime
import dateutil
import pytz


class TweetProcessor(object):
    """Class for processing tweets from a mongo database into a series of
    predictions for the sentiment, along with both the state and the time
    of a tweet

    Attributes:
    -----------
    _model: sklearn model
        model to use to predict the sentiment of the tweet.
    full_names: set
        full names of all states
    abbreviations: set
        abbreviated names of all states
    """

    def __init__(self, model, data_dir='data'):
        if isinstance(model, str):
            self.load_model(model)
        else:
            self._model = model
        states = pd.read_csv(os.path.join(data_dir, 'states.csv'))
        self.full_names = set(states.State)
        self.abbreviations = set(states.Abbreviation)
        self.state_dict = {abbreviation: state for abbreviation, state in
                           zip(states.Abbreviation, states.State)}

    def load_model(self, save_location):
        """Loads a picked model from memory and stores it
        as the self._model attribute

        Parameters:
        -----------
        save_location: str
            file location
        """
        from src.model import TweetPredictor
        with open(save_location, 'rb') as f:
            self._model = pickle.load(f)

    def find_state(self, tweet):
        """given a tweet finds the state it corresponds to
        and then returns tweet with state added in as a column

        Parameters:
        -----------
        tweet: json-like dictionary
        """
        location = tweet['user']['location']
        for state in self.full_names:
            if state.lower() in location.lower():
                return state
        parts_of_location = re.split(r'[^a-zA-Z]+', location.lower())
        for state in self.abbreviations:
            if state.lower() in parts_of_location:
                return self.state_dict[state]
        return None

    def find_sentiment(self, tweet):
        """Given a tweet finds prediction for the sentiment

        Parameters:
        -----------
        tweet: json-like dictionary

        Returns:
        --------
        sentiment, float between 0 and 1
        """
        if ("extended_tweets" in tweet) and ('full_text'
                                             in tweet['extended_tweet']):
            tweet_text = tweet['extended_tweet']['full_text']
        else:
            tweet_text = tweet['text']
        sentiment = self._model.predict_proba([tweet_text])
        return sentiment[0][1]

    @staticmethod
    def find_date(tweet):
        """finds the datetime of a tweet

        Parameters:
        -----------
        tweet: json type dictionary

        Returns:
        --------
        datetime object corresponding to when tweet was created
        """
        # return dateutil.parser.parse(tweet['created_at'])
        time = pd.Timestamp(tweet['created_at'])
        return time

    @staticmethod
    def filter_topic(tweet, topics):
        """Filter function for a tweet and list of topics, Returns True
        if the tweet contains a mention or hashtag of the topic, False
        if the tweet does not contain the topics

        Parameters:
        -----------
        tweet: json-like dict
            tweet to analyze
        topics: str or list(str)
            topic of list of topics to filter on

        Returns:
        --------
        bool, True or False depending on whether or not the tweet contains
            the topic
        """
        if topics is None:
            return True
        if ('extended_tweet' in tweet) and ('hashtags' in
                                            tweet['extended_tweet']):
            hashtags = [hashtag['text'].lower() for hashtag in
                        tweet['extended_tweet']['hashtags']]
        else:
            hashtags = [hashtag['text'].lower() for hashtag in
                        tweet['entities']['hashtags']]
        for topic in topics:
            if topic.lower() in tweet['text'].lower():
                return True
            if topic.lower() in hashtags:
                return True
        return False

    def process_predict(self, tweet):
        """Given a tweet, extracts the state, datetime, and sentiment and
        returns these in a tuple

        Parameters:
        -----------
        tweet: json-like dict
        """
        text = tweet['text']
        date = self.find_date(tweet)
        sentiment = self.find_sentiment(tweet)
        state = self.find_state(tweet)
        return text, state, date, sentiment

    def process_database(self, collection, topics=None, limit=None, n_time_bins=6):
        """Given a Pymongo connection to a mongodb collection,
        processes the tweets into a dataframe, and then returns the
        dataframe object

        Parameters:
        -----------
        collection:
            pymongo connection to mongodb table
        topics: list(str)
            list of topics to filter on, if None, there is no filter
        limit: int
            maximum number of tweets to process, if None, all tweets are
            returned
        n_time_bins: int, default 6
            number of time bins to split data into
        """
        if limit is None:
            tweets = collection.find({'user.location': {'$ne': None}})
        else:
            tweets = collection.find({'user.location': {'$ne': None}}).limit(limit)
        dataframe = pd.DataFrame([self.process_predict(tweet)
                                  for tweet in tweets if
                                  self.filter_topic(tweet, topics)],
                                 columns=['text', 'state', 'date', 'sentiment'])
        min_time = dataframe['date'].min()
        max_time = dataframe['date'].max()
        time_bin = (max_time - min_time).seconds
        dataframe['time_bin'] = dataframe['date'].map(
            lambda date: int((date - min_time).seconds/time_bin))
        return dataframe

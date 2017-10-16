"""A collection of modules to scrape tweets from a mongo database
and then process the location, and the sentiment of the tweet.
"""
import re
import pickle
import os
import pandas as pd
import dateutil


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

    def __init__(self, model, data_dir='../data'):
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
        """
        with open(save_location, 'rb') as f:
            self._model = pickle.load(f)

    def find_state(self, tweet):
        """given a tweet finds the state it corresponds to
        and then returns tweet with state added in as a column
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
        return dateutil.parser.parse(tweet['created_at'])

    def process_predict(self, tweet):
        """Given a tweet, extracts the state, datetime, and sentiment and
        returns these in a tuple
        """
        date = self.find_date(tweet)
        sentiment = self.find_sentiment(tweet)
        state = self.find_state(tweet)
        return state, date, sentiment

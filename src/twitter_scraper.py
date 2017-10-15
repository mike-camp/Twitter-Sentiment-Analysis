import requests
import datetime
from src.apikeys import TWITTER, MONGO
import tweepy
from tweepy.auth import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import Stream
from pymongo import MongoClient
US_WOEID = 23424977


def get_trends():
    """Returns a list of current trends in the US

    Parameters:
    -----------
    api: tweepy api

    Returns:
    --------
    list of trends(dictionarys), each trend has the keys:
        - name
        - promoted_content
        - query
        - tweet_volume
        - url
    """
    auth = OAuthHandler(TWITTER.CONSUMER_KEY, TWITTER.CONSUMER_SECRET)
    auth.set_access_token(TWITTER.ACCESS_TOKEN, TWITTER.ACCESS_SECRET)
    api = tweepy.API(auth)
    response = api.trends_place(US_WOEID)
    return [trend for trend in response[0]['trends']]


class CustomStreamListener(StreamListener):
    def __init__(self, table):
        super(CustomStreamListener, self).__init__()
        self.table = table

    def on_status(self, status):
        self.table.insert_one(status._json)

    def on_error(self, status_code):
        raise Exception(status_code)


def stream_trends(trend_list=None):
    """querys the list of trends and inserts them into the twitter database
    in a table with the name 'trends_<year>_<month>_<day>_<day>'

    Parameters:
    -----------
    trend_list: list of trends
    """
    if trend_list is None:
        trend_list = get_trends()
    trend_names = [trend['name'] for trend in trend_list]
    client = MongoClient('mongodb://{}:{}@localhost:27017'.format(
        MONGO.USER, MONGO.PASSWORD))
    now = datetime.datetime.now()
    table_name = 'trends_{}_{}_{}_{}'.format(now.year, now.month,
                                             now.day, now.hour)
    database = client['twitter_database']
    table = database[table_name]

    auth = OAuthHandler(TWITTER.CONSUMER_KEY, TWITTER.CONSUMER_SECRET)
    auth.set_access_token(TWITTER.ACCESS_TOKEN, TWITTER.ACCESS_SECRET)

    stream_listener = CustomStreamListener(table)
    twitter_stream = Stream(auth, stream_listener)
    twitter_stream.filter(track=trend_names)


def stream_topics(topic_list, topic_name):
    """Given a list of topics, streams the topics and then places them
    in a mongodb database in the table given by topic_name

    Parameters:
    -----------
    topic_list: list(str)
        list of topics to find tweet relating to
    topic_name: str
        name of table corresponding to topics
    """
    auth = OAuthHandler(TWITTER.CONSUMER_KEY, TWITTER.CONSUMER_SECRET)
    auth.set_access_token(TWITTER.ACCESS_TOKEN, TWITTER.ACCESS_SECRET)
    client = MongoClient('mongodb://{}:{}@localhost:27017'.format(
        MONGO.USER, MONGO.PASSWORD))
    database = client['twitter_database']

    table = database[topic_name]
    stream_listener = CustomStreamListener(table)
    twitter_stream = Stream(auth, stream_listener)
    twitter_stream.filter(track=topic_list)

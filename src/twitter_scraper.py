import requests
from apikeys import TWITTER, MONGO
import tweepy
from tweepy.auth import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import Stream
from pymongo import MongoClient
US_WOEID = 23424977


def get_trends(api):
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
    resp = api.trends_available(US_WOEID)
    return [trend for trend in resp[0]['trends']]

def stream_trends(trend_list, table):
    """querys the list of trends and inserts them into a
    database

    Parameters:
    -----------
    trend_list: list of trends
    table: connection to mongodb table
    """
    trend_names = [trend['name'] for trend in trend_list]
    stream_listener = CustomStreamListener(table)
    twitter_stream = Stream(auth, stream_listener)
    twitter_stream.filter(track=trend_names)


class CustomStreamListener(StreamListener):
    def __init__(self, table):
        super(CustomStreamListener, self).__init__()
        self.table = table

    def on_status(self, status):
        self.table.insert_one(status._json)

    def on_error(self, status_code):
        raise Exception(status_code)


if __name__=='__main__':
    auth = OAuthHandler(TWITTER.CONSUMER_KEY, TWITTER.CONSUMER_SECRET)
    auth.set_access_token(TWITTER.ACCESS_TOKEN, TWITTER.ACCESS_SECRET)
    client = MongoClient('mongodb://{}:{}@localhost:27017'.format(
        MONGO.USER, MONGO.PASSWORD))
    api = tweepy.API(auth)


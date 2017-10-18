import requests
import datetime
from src.apikeys import TWITTER, MONGO
import tweepy
import boto3
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
    def __init__(self, table, table_type='mongo', n_hours=None):
        super(CustomStreamListener, self).__init__()
        self.begin_time = datetime.datetime.now()
        self.n_hours = n_hours
        self.table_type = table_type
        self.table = table

    def on_status(self, status):
        if self.table_type == 'mongo':
            self.table.insert_one(status._json)
        elif self.table_type == 'dynamo':
            self.table.put_item(Item=status._json)
        now = datetime.datetime.now()
        # test for exit if n_hours has been set
        if self.n_hours is None:
            return True
        if (now - self.begin_time).seconds > 60*60*self.n_hours:
            return False

    def on_error(self, status_code):
        raise Exception(status_code)


def create_twitter_stream(table, n_hours=None, table_type='mongo'):
    """Creates a twitter stream object that will insert queries into
    object and will terminate in n_hours

    Parameters:
    -----------
    table: connection to mongodb table
    n_hours: number of hours to run before termination, default = None
    table_type:
        type of database to connect to.  Valid responses are:
            -mongo
            -dynamo

    Returns:
    --------
    tweepy Stream object
    """
    auth = OAuthHandler(TWITTER.CONSUMER_KEY, TWITTER.CONSUMER_SECRET)
    auth.set_access_token(TWITTER.ACCESS_TOKEN, TWITTER.ACCESS_SECRET)

    stream_listener = CustomStreamListener(table, n_hours=n_hours,
                                           table_type=table_type)
    twitter_stream = Stream(auth, stream_listener)
    return twitter_stream


def generate_mongo_table_connection(table_name):
    """Returns a connection to the twitter
    mongodb table in the twitter_database

    Parameters:
    -----------
    table_name: str
        name of table to connect to

    Returns:
    --------
    pymongo connection to the mongodb table
    """
    client = MongoClient('mongodb://{}:{}@localhost:27017'.format(
        MONGO.USER, MONGO.PASSWORD))
    database = client['twitter_database']
    return database[table_name]


def generate_dynamo_db_connection(table_name):
    dynamodb = boto3.resource('dynamodb')
    existing_tables = dynamodb.meta.client.list_tables()['TableNames']
    if table_name in existing_tables:
        return dynamodb.Table(table_name)
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName':'id',
                'KeyType':'HASH'
            },
        ],
        AttributeDefinitions=[
            {
                'AttributeName':'id',
                'AttributeType':'S'
            },
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits':1,
            'WriteCapacityUnits':1
        }
    )
    table.meta.client.get_waiter('table_exists').wait(
        TableName=table_name)
    return table


def generate_table_connection(table_name, table_type='mongo'):
    if table_type == 'mongo':
        return generate_mongo_table_connection(table_name)
    elif table_type == 'dynamo':
        return generate_dynamo_db_connection(table_name)



def generate_table_name():
    """Creates a unique table name"""
    now = datetime.datetime.now()
    table_name = 'trends_{}_{}_{}_{}'.format(now.year, now.month,
                                             now.day, now.hour)
    return table_name


def stream_trends(trend_list=None, n_trends=None, n_hours=2,
                  table_type='mongo'):
    """querys the list of trends and inserts them into the twitter database
    in a table with the name 'trends_<year>_<month>_<day>_<day>'

    Parameters:
    -----------
    trend_list: list of trends
    n_trends: number of trends to stream.
        If n_trends!=None, first n_trends of trend_list are chosen
    n_hours: number of hours to scrape trends for before exiting
    table_type: type of database to connect to
        valid responses are:
            -mongo
            -dynamo

    Returns:
    --------
    table: connection to table
    table_name: name of table in database,
    trend_names: list of names of trends
    """
    if trend_list is None:
        trend_list = get_trends()
    trend_names = [trend['name'] for trend in trend_list]
    if n_trends:
        trend_names = trend_names[:n_trends]
    table_name = generate_table_name()
    table = generate_table_connection(table_name,
                                      table_type=table_type)

    twitter_stream = create_twitter_stream(table, n_hours)
    twitter_stream.filter(track=trend_names)
    return table, table_name, trend_names


def stream_topics(topic_list, topic_name, n_hours=None,
                  table_type='mongo'):
    """Given a list of topics, streams the topics and then places them
    in a mongodb database in the table given by topic_name

    Parameters:
    -----------
    topic_list: list(str)
        list of topics to find tweet relating to
    topic_name: str
        name of table corresponding to topics
    n_hours: int
        number of hours to stream topics
    table_type: type of database to connect to
        valid responses are:
            -mongo
            -dynamo
    """
    table = generate_table_connection(topic_name,
                                      table_type=table_type)
    twitter_stream = create_twitter_stream(table, n_hours=n_hours,
                                           table_type=table_type)
    twitter_stream.filter(track=topic_list)

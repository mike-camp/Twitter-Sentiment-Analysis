from pymongo import MongoClient
from src.apikeys import TWITTER, MONGO

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


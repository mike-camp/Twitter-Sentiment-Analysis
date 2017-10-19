import time
import unittest
from src import twitter_scraper
from src.apikeys import MONGO
from pymongo import MongoClient
from multiprocessing import Process

class TestTwitterScraper(unittest.TestCase):

    def test_get_trends(self):
        resp = twitter_scraper.get_trends()
        self.assertIsInstance(resp, list)
        self.assertIsInstance(resp[0], dict)

    def test_stream_topics(self):
        client = MongoClient('mongodb://{}:{}@localhost:27017'.format(
            MONGO.USER, MONGO.PASSWORD))
        database = client['twitter_database']
        table = database['test_stream_topics']
        table.remove({})
        self.assertEqual(table.count(), 0)
        process = Process(target=twitter_scraper.stream_topics,
                          args=(['nfl', 'football'], 'test_stream_topics'))
        process.start()
        time.sleep(10)
        process.terminate()
        self.assertGreater(table.count(), 0)

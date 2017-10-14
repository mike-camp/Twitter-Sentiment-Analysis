import pickle
import unittest as unittest
from src.process_tweets import TweetProcessor


class TestTweetPreprocesor(unittest.TestCase):

    def test_retrieve_correct_state(self):
        tweet_processor = TweetProcessor(None,'data')
        with open('test/data/pickled_tweets.pk', 'rb') as f:
            tweets = pickle.load(f)
        actual_states = ['North Carolina', None, 'Texas', None, 'Virginia']
        for tweet, state in zip(tweets, actual_states):
            predicted_state = tweet_processor.find_state(tweet)
            if state is None:
                self.assertIsNone(predicted_state)
            else:
                self.assertEqual(state, predicted_state)

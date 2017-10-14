import pickle
import unittest as unittest
from src.process_tweets import TweetProcessor
from src.model import TweetPredictor


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

    def test_find_correct_date(self):
        tweet_processor = TweetProcessor(None, 'data')
        with open('test/data/pickled_tweets.pk', 'rb') as f:
            tweets = pickle.load(f)
        tweet = tweets[0]
        predicted_datetime = tweet_processor.find_date(tweet)
        self.assertEqual(predicted_datetime.month, 10)
        self.assertEqual(predicted_datetime.day, 1)
        self.assertEqual(predicted_datetime.hour, 20)
        self.assertEqual(predicted_datetime.minute, 23)

    def test_find_sentiment(self):
        # model = TweetPredictor()
        # model.train()
        # with open('models/tfidf_logistic_reg.pk', 'wb') as f:
        #     pickle.dump(model, f)
        # print('dumped model')
        with open('models/tfidf_logistic_reg.pk', 'rb') as f:
            model = pickle.load(f)
        tweet_processor = TweetProcessor(model, 'data')
        with open('test/data/pickled_tweets.pk', 'rb') as f:
            tweets = pickle.load(f)
        for tweet in tweets:
            prediction = tweet_processor.find_sentiment(tweet)
            self.assertGreaterEqual(prediction, 0.)
            self.assertLessEqual(prediction, 1.)


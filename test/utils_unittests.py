'''Unit Tests for utils functions
To run the tests: go to the root directory, Kaggle-Cancer-Dataset
run `make test`
'''
from __future__ import division
import unittest as unittest
from src import utils


class TestPreprocessing(unittest.TestCase):


    def test_remove_http_web_address(self):
        string = "check out website at https://google/docs.com/doc"
        return_string = "check out website at WEBLINK"
        result = utils.clean_text(string)
        self.assertEqual(result, return_string)


    def test_remove_www_web_address(self):
        string = "check out website at https://www.google/docs.com/doc"
        return_string = "check out website at WEBLINK"
        result = utils.clean_text(string)
        self.assertEqual(result, return_string)


    def test_remove_emoticons(self):
        string = 'this is ;( a test :< of :-)'
        return_string = 'this is  a test  of'
        emoticon_list = [';(', ':<', ':-)']
        result, emoticons = utils._remove_emoticons(string)
        self.assertEqual(result, return_string)
        for expected, result in zip(emoticon_list, emoticons):
            self.assertEqual(expected, result)


    def test_remove_mentions(self):
        string = 'hey @realdonaldtrump, what are'
        return_string = 'hey AT_USER, what are'
        result = utils.clean_text(string)
        self.assertEqual(result, return_string)

    def test_remove_hashtags(self):
        string = 'Go Buffs! #winning'
        return_string = 'Go Buffs! winning'
        result = utils.clean_text(string)
        self.assertEqual(result, return_string)



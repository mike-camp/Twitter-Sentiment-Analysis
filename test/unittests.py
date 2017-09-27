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



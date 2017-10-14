"""A module for sentiment prediction models
"""
import utils
import pickle

class TweetPredictor(object):
    """A model for predicting tweet sentiment by
    """
    def __init__(self,n_features):
        self.n_features = n_features

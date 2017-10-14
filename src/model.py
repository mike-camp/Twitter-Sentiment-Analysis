"""A module for sentiment prediction models
"""
import utils
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

class TweetPredictor(object):
    """A model for predicting tweet sentiment utilizing tfidf,
    and either logistic regression of another default predictor
    """
    def __init__(self, predictor=LogisticRegression):
        tfidf = TfidfVectorizer(stop_words='english',
                                tokenizer=utils.tokenize)
        self.pipeline = Pipeline([('tfidf', tfidf), ('predictor', predictor())])

    def load_data(self, source = 'kaggle'):
        if source == 'emoticon':
            df = pd.read_csv(
                '../data/training.1600000.processed.noemoticon.csv',
                encoding='iso8859', header=None,
                names=['sentiment', 'id', 'time', 'query', 'user', 'text'])
            df['sentiment'] = df.sentiment/4
        elif source == 'kaggle':
            df = pd.read_csv('../data/kaggle_labeled_tweets.csv',
                             delimiter='\t', header=None,
                             names=['sentiment', 'text'])
        else:
            help_string = """not a valid data source, valid sources are:
                - kaggle
                - emoticon"""
            raise Exception(help_string)
        labels = df.sentiment
        tweets = df.text
        return shuffle(tweets, labels)

    def train(self, verbose=False, source='kaggle'):
        params = {'tfidf__min_df':[.01, .05, .1],
                  'tfidf__max_df':[1., .9, .8, .7],
                  'predictor__C':[2**n for n in range(-3, 3)]}
        grid_search = GridSearchCV(self.pipeline, params,
                                   scoring='neg_log_loss',
                                   n_jobs=-1)
        tweets, labels = self.load_data(source)
        grid_search.fit(tweets, labels)
        self.grid_search = grid_search
        if verbose:
            cv_res = grid_search.cv_results_
            print('           :   C   : min_df  : max_df  ')
            for score,std,params in zip(cv_res['mean_test_score'],
                                        cv_res['std_test_score'],
                                        cv_res['params']):
                print('{:.3f}+/-{:3f} : {} : {} : {}'.format(
                        score,std,params['predictor__C'],
                        params['tfidf__min_df'],
                        params['tfidf__max_df']))
            print(grid_search.best_params_)
            print(grid_search.best_score_)
        self._estimator = grid_search.best_estimator_

    def predict_proba(self, tweets):
        if self._estimator is None:
            raise Exception('model must be trained before prediction')
        return self._estimator.predict_proba(tweets)

if __name__=='__main__':
    model = TweetPredictor()
    model.train(verbose=True, source='kaggle')


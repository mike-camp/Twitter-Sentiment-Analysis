"""A module for sentiment prediction models
"""
from src import utils
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

    Attributes:
    -----------
    predictor: sklearn estimator
        model to use for predicting probabilities after TFIDF
    pipeline: sklearn pipeline
        pipeline to transform words to probability predictions
    _grid_search: sklearn GridSearchCV
        grid search to find best parameters
    _best_estimator:
        best estimator found by grid_search
    """
    def __init__(self, predictor=LogisticRegression()):
        self.predictor = predictor
        tfidf = TfidfVectorizer(stop_words='english',
                                tokenizer=utils.tokenize)
        self.pipeline = Pipeline([('tfidf', tfidf), ('predictor', predictor)])

    def load_data(self, source='kaggle'):
        """Loads the data in, processes it into a standard form
        with sentiment labeled as 0 or 1

        Parameters:
        -----------
        source: str, 'kaggle' or 'emoticon'
            data source to use for training

        Returns:
        --------
        tweets: list(str)
            list of text of tweets
        labels: list(int)
            list of sentiment labels (0=negative, 1=positive)
        """
        if source == 'emoticon':
            df = pd.read_csv(
                'data/training.1600000.processed.noemoticon.csv',
                encoding='iso8859', header=None,
                names=['sentiment', 'id', 'time', 'query', 'user', 'text'])
            df['sentiment'] = df.sentiment/4
        elif source == 'kaggle':
            df = pd.read_csv('data/kaggle_labeled_tweets.csv',
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
        """uses the loaded data to run a grid search for the best
        parameters, and then stores the best estimator as an
        instance variable.

        Parameters:
        ----------
        verbose: bool
        source: str, options = ['kaggle','emoticon']
            source of tweets and labels to use
        """
        params = {'tfidf__min_df': [.01, .05, .1],
                  'tfidf__max_df': [1., .9, .8, .7],
                  'predictor__C': [2**n for n in range(-3, 3)]}
        grid_search = GridSearchCV(self.pipeline, params,
                                   scoring='neg_log_loss',
                                   n_jobs=-1)
        tweets, labels = self.load_data(source)
        grid_search.fit(tweets, labels)
        self.grid_search = grid_search
        if verbose:
            cv_res = grid_search.cv_results_
            print('           :   C   : min_df  : max_df  ')
            for score, std, params in zip(cv_res['mean_test_score'],
                                          cv_res['std_test_score'],
                                          cv_res['params']):
                print('{:.3f}+/-{:3f} : {} : {} : {}'.format(
                        score, std, params['predictor__C'],
                        params['tfidf__min_df'],
                        params['tfidf__max_df']))
            print(grid_search.best_params_)
            print(grid_search.best_score_)
        self._estimator = grid_search.best_estimator_
        self._estimator.fit(tweets,labels)

    def predict_proba(self, tweets):
        """Given a collection of tweets, predicts the probability
        of a positive sentiment for all tweets

        Parameters:
        -----------
        tweets: list(str)
            list of tweets to predict sentiment for

        Returns:
        --------
        sentiments: array, shape = [n_tweets,2]
            predicted probabilities for positive and negative sentiments
        """

        if self._estimator is None:
            raise Exception('model must be trained before prediction')
        return self._estimator.predict_proba(tweets)

if __name__ == '__main__':
    model = TweetPredictor()
    with open('models/untrained_emoticon_lr_model.pk','wb') as f:
        pickle.dump(model,f)
    model.train(verbose=True, source='emoticon')
    with open('models/emoticon_lr_model.pk','wb') as f:
        pickle.dump(model,f)

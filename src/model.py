"""A module for sentiment prediction models
"""
from src import utils
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation


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
    def __init__(self, model_type):
        self.model_type = model_type
        tfidf = TfidfVectorizer(stop_words='english',
                                tokenizer=utils.tokenize)
        self.pipeline = self.create_pipeline(model_type)
        self.param_grid = self.create_param_grid(model_type)

    def create_pipeline(self, model_type):
        if model_type=='stemmed_lr':
            tfidf = TfidfVectorizer(stop_words='english',
                                    tokenizer=utils.tokenize_and_stem)
            predictor = LogisticRegression()
        elif model_type=='unstemmed_lr':
            tfidf = TfidfVectorizer(stop_words='english',
                                    tokenizer=utils.tokenize)
            predictor = LogisticRegression()
        elif model_type == 'stemmed_rf':
            tfidf = TfidfVectorizer(stop_words='english',
                                    tokenizer=utils.tokenize_and_stem)
            predictor = RandomForestClassifier(n_estimators=10000)
        elif model_type == 'stemmed_nmf_lr':
            return self.create_three_stage_pipeline(model_type)
        elif model_type == 'stemmed_nmf_rf':
            return self.create_three_stage_pipeline(model_type)
        elif model_type == 'stemmed_lda_lr':
            return self.create_three_stage_pipeline(model_type)
        elif model_type == 'unstemmed_lda_rf':
            return self.create_three_stage_pipeline(model_type)

        pipeline = Pipeline([('tfidf', tfidf), ('predictor', predictor)])
        return pipeline

    def create_three_stage_pipeline(self, model_type):
        tfidf = TfidfVectorizer(stop_words='english',
                                tokenizer=utils.tokenize_and_stem)
        if model_type == 'stemmed_nmf_lr':
            feature_reducer = NMF()
            predictor = LogisticRegression()
        elif model_type == 'stemmed_nmf_rf':
            feature_reducer = NMF()
            predictor = RandomForestClassifier(n_estimators=10000)
        elif model_type == 'stemmed_lda_lr':
            feature_reducer = LatentDirichletAllocation()
            predictor = LogisticRegression()
        elif model_type == 'stemmed_lda_rf':
            feature_reducer = LatentDirichletAllocation()
            predictor = RandomForestClassifier(n_estimators=10000)
        pipeline = Pipeline([('tfidf', tfidf),
                             ('feature_reducer', feature_reducer),
                             ('predictor', predictor)])
        return pipeline

    def create_param_grid(self, model_type):
        if model_type == 'stemmed_lr' or model_type == 'unstemmed_lr':
            param_grid = {'tfidf__min_df': [.01, .05, .1],
                          'tfidf__max_df': [1., .9, .8, .7],
                          'predictor__C': [2**n for n in range(-3, 3)]}
        elif model_type == 'stemmed_rf':
            param_grid = {'tfidf__min_df': [.01, .05, .1],
                          'tfidf__max_df': [1., .9, .8, .7],
                          'predictor__max_depth': [None, 3, 5, 7]}
        elif model_type == 'stemmed_nmf_lr' or model_type == 'stemmed_lda_lr':
            param_grid = {'feature_reducer__n_components':[100, 200,
                                                           500, 1000],
                          'predictor__C': [2**n for n in range(-1, 4, 2)]}
        elif model_type == 'stemmed_nmf_rf' or model_type == 'stemmed_lda_rf':
            param_grid = {'feature_reducer__n_components':[100, 200,
                                                           500, 1000],
                          'predictor__max_depth': [None, 3, 5, 7]}
        return param_grid

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

    def train(self, source='kaggle'):
        """uses the loaded data to run a grid search for the best
        parameters, and then stores the best estimator as an
        instance variable.

        Parameters:
        ----------
        verbose: bool
        source: str, options = ['kaggle','emoticon']
            source of tweets and labels to use
        """
        grid_search = GridSearchCV(self.pipeline, self.param_grid,
                                   scoring='neg_log_loss',
                                   n_jobs=-1)
        tweets, labels = self.load_data(source)
        grid_search.fit(tweets, labels)
        self.grid_search = grid_search
        cv_res = grid_search.cv_results_
        results = '           :   C   : min_df  : max_df  '
        for score, std, params in zip(cv_res['mean_test_score'],
                                      cv_res['std_test_score'],
                                      cv_res['params']):
            results += '\n{:.3f}+/-{:3f} : {} : {} : {}'.format(
                        score, std, params['predictor__C'],
                        params['tfidf__min_df'],
                        params['tfidf__max_df'])
            print(results.split('\n')[-1])
        results += '\n\ncv_best results:'
        results += '{}'.format(grid_search.best_params_)
        results += '{}'.format(grid_search.best_score_)

        with open('cv_results_{}.txt'.format(self.model_type), 'w') as f:
            f.write(results)
        print(grid_search.best_params_)
        print(grid_search.best_score_)
        self._estimator = grid_search.best_estimator_
        self._estimator.fit(tweets, labels)

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
    model = TweetPredictor('stemmed_lr')
    with open('models/untrained_emoticon_lr_model.pk','wb') as f:
        pickle.dump(model,f)
    model.train(verbose=True, source='emoticon')
    with open('models/emoticon_lr_model.pk','wb') as f:
        pickle.dump(model,f)

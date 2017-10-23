"""A collection of neural network implementations
for classifying sentiment
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from src import utils
from sklearn.utils import shuffle
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import multiprocessing

class RNN(object):
    """A class for creating a character level
    recurrent neural network
    """

    def __init__(self, n_epochs, embedding_dim=126, batch_size=100,
                 cell_type='rnn', n_hidden=100, n_classes=2,
                 learning_rate=.01, max_sequence_length=20,
                 l2=.01):
        self.n_epochs = n_epochs
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.cell_type='rnn'
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.max_sequence_length = max_sequence_length
        self.lr = learning_rate
        self.l2 = l2
        tweets, lengths, labels = preprocess_data(embedding_dim,
                                                  max_sequence_length)
        self.tweets = np.array(tweets)
        self.lengths = np.array(lengths)
        self.labels = np.array(labels)



    def split(self, ls, sublist_length=2000):
        """shuffles, then splits a list into several chunks of length
        sublist_length

        Parameters:
        -----------
        ls: iterable to be split

        Returns:
        --------
        iterable of sublists
        """
        return [ls[i:i+sublist_length] for i in
                range(0, len(ls), sublist_length)]

    def partial_fit(self, sess, tweets, lengths, labels):
        """partially fits a neural network to classify tweets

        Parameters:
        ----------
        tweets, list(str)
            list of tweets to classify
        labels, list(int)
            list of tweet labels
        """
        for batch_x, batch_length, batch_label in\
                             self._get_mini_batches(tweets,
                                                    lengths,
                                                    labels):
            feed_dict = {self.x_input:batch_x,
                         self.length_input:batch_length,
                         self.label_input:batch_label}
            sess.run((self.optimizer, self.accuracy_update,
                      self.precision_update,
                      self.recall_update), feed_dict=feed_dict)

    def fit(self, tweets, labels):
        """Fits a neural network to classify tweets

        Paramters:
        ----------
        tweets, list(str)
            list of tweets to classify
        labels, list(int)
            list of tweet labels
        """
        self._create_placeholders()
        indices = np.arange(len(tweets))
        labels = np.array(labels)

        logit_predictions = self._get_logit_predictions(self.x_input, self.length_input)
        one_hot_labels = tf.one_hot(indices=self.label_input, depth=self.n_classes,
                                    on_value=1, off_value=0, dtype=tf.int32)
        self.cost, self.optimizer = self._optimize(logit_predictions, one_hot_labels)
        self.predictions = tf.argmax(logit_predictions, 1)

        with tf.name_scope('measurements'):
            precision, self.precision_update =\
                tf.contrib.metrics.streaming_precision(self.predictions, self.label_input)
            recall, self.recall_update = \
                tf.contrib.metrics.streaming_recall(self.predictions, self.label_input)
            accuracy, self.accuracy_update = \
                tf.contrib.metrics.streaming_accuracy(self.predictions, self.label_input)

        # Store variables related to metrics in a list which allows all 
        # measurement variables to be reset
        stream_vars = [i for i in tf.local_variables() if
                        i.name.split('/')[0] == 'measurements']
        reset_measurements = [tf.variables_initializer(stream_vars)]
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('running')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for epoch in range(self.n_epochs):
                for split in self.split(indices):
                    self.partial_fit(sess, tweets[split], labels[split])
                a, p, r = sess.run((accuracy, precision, recall))
                sess.run(reset_measurements)
                f1_score = 2*p*r/(p+r)
                print("testing accuracy={:.4f}, f1={:.5f}"\
                      .format(a, f1_score))
                saver.save(sess,"checkpoints/model.ckpt")


    def predict(self,tweets):
        """Takes a collections of tweets, processes them, and then
        finds the predictions.

        Parameters:
        -----------
        tweets:
            list of tweets to classify

        Returns:
        --------
        Predicted sentiment of tweets
        """
        tweets, lengths = utils._encode_tweet_collection(tweets,
                                                self.max_sequence_length,
                                                self.embedding_dim)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,"checkpoints/model.ckpt")

            feed_dict = {self.x_input:tweets,
                         self.length_input:lengths}
            predictions = sess.run(self.predictions, feed_dict=feed_dict)
            return predictions



    def score(self, tweets, predictions):

        tweets, lengths = utils._encode_tweet_collection(tweets,
                                                         self.max_sequence_length,
                                                         self.embedding_dim)
        labels = np.array(predictions)
        feed_dict = {self.x_input:tweets,
                     self.length_input:lengths,
                     self.label_input:labels}
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,"checkpoints/model.ckpt")
            cost = sess.run(self.cost,feed_dict=feed_dict)
        return cost

    def _create_placeholders(self):
        self.x_input = tf.placeholder(tf.float32,
                                      shape=[None,
                                             self.max_sequence_length,
                                             self.embedding_dim])
        self.length_input = tf.placeholder(tf.int32, shape=[None])
        self.label_input = tf.placeholder(tf.int32, shape=[None])

    def _get_mini_batches(self, X, lengths, labels):
        """Creates a list of minibatches of the training data and labels
        """
        return [(X[i:i+self.batch_size],
                 lengths[i:i+self.batch_size],
                 labels[i:i+self.batch_size])
                for i in range(0, len(X), self.batch_size)]

    def _get_logit_predictions(self, X, lengths, reuse_variables=False):
        """Returns the logits for each vectorized tweet in
        an array.

        Parameters:
        -----------
        X : arraylike shape = [n_samples,num_chars,embedding_dim]
            the vectorized tweets
        lengths: arraylike, shape = [n_samples]
            the lengths of the tweets

        Returns:
        --------
        logits : shape=[n_samples,n_classes]
            logits corresponding to different class predictions for
        """
        with tf.variable_scope('RNN', reuse=reuse_variables):
            if self.cell_type == 'rnn':
                cell = tf.contrib.rnn.BasicRNNCell(self.n_hidden,
                                                   reuse=reuse_variables)
            elif self.cell_type == 'lstm':
                cell = tf.contrib.rnn.BasicLSTMCEll(self.n_hidden,
                                                    reuse=reuse_variables,
                                                    forget_bias=.6)
            elif self.cell_type == 'gru':
                cell = tf.contrib.rnn.BasicGRUCell(self.n_hidden,
                                                   reuse=reuse_variables)
            outputs, states = tf.nn.dynamic_rnn(cell,
                                                inputs=X,
                                                sequence_length=lengths,
                                                time_major=False,
                                                dtype=tf.float32)
            W = tf.get_variable('W', shape=[self.n_hidden, self.n_classes])
            biases = tf.get_variable('b', shape=[self.n_classes])
        #outputs has shape 100,140,100
        output_list = tf.unstack(outputs, self.max_sequence_length, 1)
        #output_list has shape 140,100,100
        batch_range = tf.range(tf.shape(outputs)[0])
        indices = tf.stack([batch_range, lengths-1], axis=1)
        # tf.stack([...],axis=1) has shape 100,2
        # last_rnn_output = tf.gather_nd(outputs, tf.stack(
        #     [tf.range(batch_size), lengths-1], axis=1))
        ### THE ERROR IS ON THE NEXT LINE< TF.GATHER_ND CHANGED
        ### FROM TENSORFLOW 1.1 to TENSORFLOW 1.3
        last_rnn_output = tf.gather_nd(outputs, indices)

        return tf.matmul(last_rnn_output, W)+biases


    def _get_cost(self, pred, labels):
        """Finds the cross entropy loss given a list of logits

        Parameters:
        -----------
        pred : arraylike, shape=[n_samples,n_classes]
            logit predictions
        labels : arraylike, shape = [n_samples]
            actual labels

        Returns:
        --------
        cross entropy cost
        """

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pred, labels=labels))
        return cost

    def _optimize(self, pred, labels):
        """Finds the cross entropy loss given a list of logits

        Parameters:
        -----------
        pred : arraylike, shape=[n_samples,n_classes]
            logit predictions
        labels : arraylike, shape = [n_samples]
            actual labels

        Returns:
        --------
        cross entropy cost
        """

        cost = self._get_cost(pred, labels)
        regulated_cost = cost
        with tf.variable_scope('RNN', reuse=True):
            regulated_cost += self.l2*tf.nn.l2_loss(tf.get_variable('W'))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)\
                .minimize(regulated_cost)
        return cost, optimizer


def preprocess_data_from_mongo(X, embedding_dim=300, sentence_length=20):
    """Loads the text data from a  mongodb connection and then
    processes it into a padded array, which is then returned
    along with an array corresponding to sequence lengths
    """
    pass


def preprocess_data(embedding_dim=10, sentence_length=10, n_examples = None):
    """Loads the data from the emoticon labeled tweets, and then processes
    these using word to vec

    Parameters:
    -----------
    embedding_dim: int
        size of vectors to embed words in
    sentence_length: int
        maximum length of sentence.  Sentences shorter than this length
        will be padded.  Longer sentences will be shortened.

    Returns:
    --------
    tokenized_tweets: arraylike, shape=[n_tweets,sentence_length,embedding_dim]
        embedding of tweets
    labels: arraylike, shape=[n_tweets,num_classes]
    lengths: arraylike, shape=[n_tweets]
        lengths of tweets before processing, tweets longer than
        sentence_length are shortened to sentence_length
    """
    dataframe = pd.read_csv('../data/training.1600000.processed.noemoticon.csv',
                            encoding='iso8859', header=None,
                            names=['sentiment', 'id', 'time', 'query', 'user',
                                   'text'])
    if n_examples:
        dataframe = dataframe.iloc[:n_examples, :]
    labels = dataframe.sentiment/4
    tweets = dataframe.text
    tweets, labels = shuffle(tweets, labels)
    # tokenized_tweets = [utils.tokenize_and_stem(tweet) for tweet in tweets]
    pool = multiprocessing.pool.Pool()
    tokenized_tweets = pool.map(utils.tokenize_and_stem, tweets)
    pool.close()
    pool.join()
    word_to_vec = Word2Vec(tokenized_tweets, size=embedding_dim).wv
    print('tokenized_tweets')
    tokenized_tweets = [[list(word_to_vec[word]) for word in tweet if
                         word in word_to_vec.vocab] for tweet in
                        tokenized_tweets]
    lengths = [len(tweet) for tweet in tokenized_tweets]
    lengths = [min(length, sentence_length) for length in lengths]
    pad = [0. for i in range(embedding_dim)]
    tokenized_tweets = [[tweet[i] if i < len(tweet) else pad for i in
                         range(sentence_length)] for tweet in
                        tokenized_tweets]
    tokenized_tweets = [tweet for tweet, length in zip(tokenized_tweets, lengths)
                        if length > 0]
    labels = [label for label, length in zip(labels, lengths)
              if length > 0]
    lengths = [length for length in lengths if length > 0]
    labels = [[0, 1] if label else [1, 0] for label in labels]
    return tokenized_tweets, labels, lengths




class CNN(object):
    """ A CNN for tweet classification,
    Uses word2vec as the initial layer, followed by a convolutional
    layer

    Taken from wildml.com/2015/12/implementing-a-cnn-for-text\
            -classification-in-tensorflow

    Attributes:
    -----------
    sequence_length: int, length of sentences
    num_classes: int, number of classes to predict
    embedding_size, int, dim of word2vec word embedding
    filter_sizes: list(int), number of words we want each of
        the convolutional filters to cover
    num_filters: list(int), number of filters per filter size
    """
    def __init__(self, sequence_length, num_classes, embedding_size,
                 filter_sizes, num_filters, learning_rate=.001,
                 batch_size=100, num_epochs=1000,
                 dropout_keep_prob=.6, store_data=True):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.learning_rate= learning_rate
        self.x_input = tf.placeholder(tf.float32,
                                      shape=[None, sequence_length, embedding_size],
                                      name='x_input')
        self.y_input = tf.placeholder(tf.float32,
                                      [None, num_classes],
                                      name='y_input')
        self.length_input = tf.placeholder(tf.int32,
                                           [None],
                                           name='length_input')
        self.filter_sizes = filter_sizes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.embedding_size = embedding_size
        self.dropout_keep_prob = dropout_keep_prob
        self.store_data = store_data
        self.create_word_dict()
        self.setup_graph()

    def create_word_dict(self):
        """Loads the data from the emoticon labeled tweets, and then processes
        these using word to vec

        Parameters:
        -----------
        embedding_dim: int
            size of vectors to embed words in
        sentence_length: int
            maximum length of sentence.  Sentences shorter than this length
            will be padded.  Longer sentences will be shortened.

        Returns:
        --------
        tokenized_tweets: arraylike, shape=[n_tweets,sentence_length,embedding_dim]
            embedding of tweets
        labels: arraylike, shape=[n_tweets,num_classes]
        lengths: arraylike, shape=[n_tweets]
            lengths of tweets before processing, tweets longer than
            sentence_length are shortened to sentence_length
        """
        dataframe = pd.read_csv('../data/training.1600000.processed.noemoticon.csv',
                                encoding='iso8859', header=None,
                                names=['sentiment', 'id', 'time', 'query', 'user',
                                       'text'])
        labels = dataframe.sentiment/4
        tweets = dataframe.text
        tweets, labels = shuffle(tweets, labels)
        # tokenized_tweets = [utils.tokenize_and_stem(tweet) for tweet in tweets]
        pool = multiprocessing.pool.Pool()
        tokenized_tweets = pool.map(utils.tokenize_and_stem, tweets)
        pool.close()
        pool.join()

        lengths = [len(tweet) for tweet in tokenized_tweets]
        lengths = [min(length, self.sequence_length) for length in lengths]

        tokenized_tweets = [tweet for tweet, length in zip(tokenized_tweets, lengths)
                            if length > 0]
        labels = [label for label, length in zip(labels, lengths)
                  if length > 0]
        lengths = [length for length in lengths if length > 0]


        word_to_vec = Word2Vec(tokenized_tweets, size=self.embedding_size).wv
        self.word_to_vec = word_to_vec
        self.tweets = tweets
        self.labels = labels
        return

    def encode_tweets(self,tweets):
        tokenized_tweets = [[list(self.word_to_vec[word]) for word in tweet if
                             word in self.word_to_vec.vocab] for tweet in
                            tweets]
        pad = [0. for i in range(self.embedding_size)]
        tokenized_tweets = [[tweet[i] if i < len(tweet) else pad for i in
                             range(self.sequence_length)] for tweet in
                            tokenized_tweets]
        return tokenized_tweets

    def encode_labels(self, labels):
        labels = [[0, 1] if label else [1, 0] for label in labels]
        return labels

    def create_one_conv(self, x_input, filter_size):
        filter_shape = [filter_size, self.embedding_size, 1,
                        self.num_filters]
        with tf.variable_scope('conv_filter_{}'.format(filter_size),
                               reuse=False):
            W = tf.get_variable('W', filter_shape, dtype=tf.float32)
            b = tf.get_variable('b', [self.num_filters], dtype=tf.float32)
        conv = tf.nn.conv2d(x_input, W, strides=[1, 1, 1, 1],
                            padding='VALID', name='conv')
        h = tf.nn.relu(conv+b)
        pooled = tf.nn.max_pool(h, ksize=[1,
                                          self.sequence_length-filter_size+1,
                                          1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name='pool')
        return pooled

    def create_conv_layer(self, x_input):
        pooled_outputs = list()
        for filter_size in self.filter_sizes:
            pooled = self.create_one_conv(x_input, filter_size)
            pooled_outputs.append(pooled)
        pooled_outputs = tf.concat(pooled_outputs, 3)
        num_filters_total = self.num_filters * len(self.filter_sizes)
        flattened_pool = tf.reshape(pooled_outputs, [-1, num_filters_total])
        return flattened_pool

    def add_dropout(self, x_input):
        num_filters_total = self.num_filters * len(self.filter_sizes)
        dropped_x = tf.nn.dropout(x_input, self.dropout_keep_prob)
        return dropped_x

    def create_predictions(self, x_input, reuse):
        num_filters_total = self.num_filters * len(self.filter_sizes)
        with tf.variable_scope('output', reuse=reuse):
            W = tf.get_variable('W', shape=[num_filters_total,
                                            self.num_classes])
            b = tf.get_variable('b', shape=[self.num_classes])
        predictions = tf.matmul(x_input, W)+b
        return predictions

    def create_convolutional_graph(self, x_input):
        expanded_x = tf.expand_dims(x_input, -1)
        pooled_outputs = self.create_conv_layer(expanded_x)
        dropout_pooled_outputs = self.add_dropout(pooled_outputs)
        self.training_prob = self.create_predictions(dropout_pooled_outputs,
                                                     False)
        self.test_prob = self.create_predictions(pooled_outputs,True)
        self.training_predictions = tf.argmax(self.training_prob, axis=1)
        self.test_predictions = tf.argmax(self.test_prob, axis=1)

    def create_loss_function(self, pred_prob, labels):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=pred_prob,
                                                         labels=labels)
        return tf.reduce_mean(losses)

    def create_training_function(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                .minimize(loss)
        return optimizer

    def setup_graph(self):
        self.create_convolutional_graph(self.x_input)
        self.training_loss = self.create_loss_function(self.training_prob,
                                                       self.y_input)
        self.test_loss = self.create_loss_function(self.test_prob,
                                                   self.y_input)
        self.train_op = self.create_training_function(self.training_loss)

    def minibatches(self, X, y):
        for i in range(len(X)//self.batch_size):
            lower = i*self.batch_size
            upper = (i+1)*self.batch_size
            yield X[lower:upper], y[lower:upper]

    def run_training_epoch(self, sess, X, y):
        X, y = shuffle(X, y)
        losses = []
        for batch_x, batch_y in self.minibatches(X, y):
            batch_x = self.encode_tweets(batch_x)
            batch_y = self.encode_labels(batch_y)
            feed_dict = {self.x_input:batch_x,
                         self.y_input:batch_y}
            loss, _ = sess.run((self.training_loss, self.train_op),
                               feed_dict=feed_dict)
            losses.append(loss)
        return np.mean(losses)

    def evaluate_test_set(self, sess, X, y):
        losses = []
        for batch_x, batch_y in self.minibatches(X, y):
            batch_x = self.encode_tweets(batch_x)
            batch_y = self.encode_labels(batch_y)
            feed_dict = {self.x_input:batch_x,
                         self.y_input:batch_y}
            loss, _ = sess.run(self.test_loss,
                               feed_dict=feed_dict)
            losses.append(loss)
        return np.mean(losses)

    def train(self):
        """Trains the graph using holdout validation to ensure that the test
        error is decreasing
        """
        X_train, X_test, y_train, y_test = train_test_split(self.tweets,
                                                            self.labels,
                                                            test_size=.2)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_loss = 1000
            for i in range(self.num_epochs):
                training_loss = self.run_training_epoch(sess, X_train,
                                                        y_train)
                current_test_loss = self.evaluate_test_set(sess, X_test,
                                                           y_test)
                if current_test_loss < test_loss:
                    test_loss = current_test_loss
                    saver.save(sess, 'checkpoints/cnn.ckpt')
                training_string = ('epoch: {} -- '+
                                   'training loss:{:.3f} -- '+
                                   'test loss: {:.3f}')
                print(training_string.format(i, training_loss, test_loss))

    def predict_proba(self, X):
        """predicts the probability of positive or negative
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, 'checkpoints/cnn.ckpt')
            feed_dict = {self.x_input:X}
            prob = sess.run(tf.nn.softmax(self.test_prob),
                            feed_dict=feed_dict)
        return prob

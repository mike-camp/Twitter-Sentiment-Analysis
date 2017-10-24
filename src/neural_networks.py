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
    """A class for creating a simple word2vec based
    recurrent neural network
    """

    def __init__(self, num_epochs, tokenization='word', embedding_dim=300,
                 batch_size=100, cell_type='rnn', n_hidden=100,
                 num_classes=2, learning_rate=.01, max_sequence_length=20,
                 l2=.001):
        self.num_epochs = num_epochs
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.cell_type='rnn'
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.max_sequence_length = max_sequence_length
        self.learning_rate = learning_rate
        self.l2 = l2
        self.tokenization = tokenization
        if tokenization=='word':
            self.x_input = tf.placeholder(tf.float32,
                                          shape=[None, max_sequence_length,
                                                 embedding_dim],
                                          name='x_input')
        elif tokenization=='char':
            self.x_input = tf.placeholder(tf.float32,
                                          shape=[None, max_sequence_length,
                                                 1],
                                          name='x_input')
        self.y_input = tf.placeholder(tf.float32,
                                      [None, num_classes],
                                      name='y_input')
        self.length_input = tf.placeholder(tf.int32,
                                           [None],
                                           name='length_input')

        if tokenization=='word':
            self.create_word_dict()
            self.encode_tweets = self.encode_tweets_word2vec
        elif tokenization=='char':
            self.load_tweets_char_level()
            self.encode_tweets = self.encode_tweets_char_level
        else:
            raise Exception('Invalid tokenization parameter,'+
                            'valid options are word or char')
        self.setup_graph()

    def load_tweets_char_level(self):
        dataframe = pd.read_csv('../data/training.1600000.processed.noemoticon.csv',
                                encoding='iso8859', header=None,
                                names=['sentiment', 'id', 'time', 'query', 'user',
                                       'text'])
        labels = dataframe.sentiment/4
        tweets = dataframe.text
        tweets, labels = shuffle(tweets, labels)
        tweets = [tweet[:self.max_sequence_length] for tweet in tweets]
        lengths = [len(tweet) for tweet in tweets]

        tokenized_tweets = [tweet for tweet, length in zip(tweets, lengths)
                            if length > 0]
        labels = [label for label, length in zip(labels, lengths)
                  if length > 0]
        lengths = [length for length in lengths if length > 0]

        self.tweets = tweets
        self.labels = labels
        self.lengths = lengths
        return

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
        lengths = [min(length, self.max_sequence_length) for length in lengths]

        tokenized_tweets = [tweet for tweet, length in zip(tokenized_tweets, lengths)
                            if length > 0]
        labels = [label for label, length in zip(labels, lengths)
                  if length > 0]
        lengths = [length for length in lengths if length > 0]

        word_to_vec = Word2Vec(tokenized_tweets, size=self.embedding_dim).wv
        self.word_to_vec = word_to_vec
        self.tweets = tweets
        self.labels = labels
        self.lengths = lengths
        return

    def encode_tweets_char_level(self, tweets):
        def encode(letter):
            if ord(letter) < self.embedding_dim:
                return [ord(letter)]
            return [0]
        vectorized_tweets = [[encode(tweet[i]) if i < len(tweet) else [0]
                              for i in range(self.max_sequence_length)]
                             for tweet in tweets]
        return vectorized_tweets

    def encode_tweets_word2vec(self,tweets):
        tokenized_tweets = [[list(self.word_to_vec[word]) for word in tweet if
                             word in self.word_to_vec.vocab] for tweet in
                            tweets]
        pad = [0. for i in range(self.embedding_dim)]
        tokenized_tweets = [[tweet[i] if i < len(tweet) else pad for i in
                             range(self.max_sequence_length)] for tweet in
                            tokenized_tweets]
        return tokenized_tweets

    def encode_labels(self, labels):
        labels = [[0, 1] if label else [1, 0] for label in labels]
        return labels

    def create_recurrent_layer(self, X, lengths):
        with tf.variable_scope('RNN'):
            if self.cell_type == 'rnn':
                cell = tf.contrib.rnn.BasicRNNCell(self.n_hidden)
            elif self.cell_type == 'lstm':
                cell = tf.contrib.rnn.BasicLSTMCEll(self.n_hidden,
                                                    forget_bias=.6)
            elif self.cell_type == 'gru':
                cell = tf.contrib.rnn.BasicGRUCell(self.n_hidden)
            outputs, states = tf.nn.dynamic_rnn(cell,
                                                inputs=X,
                                                sequence_length=lengths,
                                                time_major=False,
                                                dtype=tf.float32)
        batch_range = tf.range(tf.shape(outputs)[0])
        indices = tf.stack([batch_range, lengths-1], axis=1)
        last_rnn_output = tf.gather_nd(outputs, indices)
        return last_rnn_output

    def add_full_layer(self, X):
        with tf.variable_scope('full_1'):
            W = tf.get_variable('W', shape=[self.n_hidden, self.num_classes])
            b = tf.get_variable('b', shape=[self.num_classes])
        return tf.matmul(X, W)+b

    def get_training_cost(self, logits, labels):
        costs = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=labels)
        cost = tf.reduce_mean(costs)
        with tf.variable_scope('full_1', reuse=True):
            W = tf.get_variable('W')
        cost += self.l2*tf.nn.l2_loss(W)
        return tf.reduce_mean(costs)

    def get_test_cost(self, logits, labels):
        costs = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=labels)
        return tf.reduce_mean(costs)

    def get_training_operation(self, cost):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        loss_operation = optimizer.minimize(cost)
        return loss_operation

    def setup_graph(self):
        if self.tokenization=='word':
            x_input = self.x_input
        elif self.tokenization=='char':
            identity = tf.constant(np.eye(self.embedding_dim))
            x_input = tf.gather_nd(identity, self.x_input)
        rnn_outputs = self.create_recurrent_layer(x_input,
                                                  self.length_input)
        self.logits = self.add_full_layer(rnn_outputs)
        self.training_cost = self.get_training_cost(self.logits,
                                                    self.y_input)
        self.test_cost = self.get_test_cost(self.logits,
                                            self.y_input)
        self.train_op = self.get_training_operation(self.test_cost)
        self.probs = tf.nn.softmax(self.logits)

    def minibatches(self, X, y, length):
        for i in range(len(X)//self.batch_size):
            lower = i*self.batch_size
            upper = (i+1)*self.batch_size
            yield X[lower:upper], y[lower:upper], length[lower:upper]

    def run_training_epoch(self, sess, X, y, length):
        X, y = shuffle(X, y)
        losses = []
        for batch_x, batch_y, batch_length in self.minibatches(X, y, length):
            batch_x = self.encode_tweets(batch_x)
            batch_y = self.encode_labels(batch_y)
            feed_dict = {self.x_input:batch_x,
                         self.y_input:batch_y,
                         self.length_input:batch_length}
            loss, _ = sess.run((self.training_cost, self.train_op),
                               feed_dict=feed_dict)
            losses.append(loss)
        return np.mean(losses)

    def evaluate_test_set(self, sess, X, y, length):
        X, y = shuffle(X, y)
        losses = []
        for batch_x, batch_y, batch_length in self.minibatches(X, y, length):
            batch_x = self.encode_tweets(batch_x)
            batch_y = self.encode_labels(batch_y)
            feed_dict = {self.x_input:batch_x,
                         self.y_input:batch_y,
                         self.length_input:batch_length}
            loss = sess.run(self.test_cost,
                               feed_dict=feed_dict)
            losses.append(loss)
        return np.mean(losses)

    def train(self):
        """Trains the graph using holdout validation to ensure that the test
        error is decreasing
        """
        X_train, X_test, y_train, y_test, length_train, length_test=\
            train_test_split(self.tweets, self.labels, self.lengths,
                             test_size=.2)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_loss = 1000
            for i in range(self.num_epochs):
                training_loss = self.run_training_epoch(sess, X_train,
                                                        y_train,
                                                        length_train)
                current_test_loss = self.evaluate_test_set(sess, X_test,
                                                           y_test,
                                                           length_test)
                if current_test_loss < test_loss:
                    test_loss = current_test_loss
                    saver.save(sess, 'checkpoints/rnn.ckpt')
                training_string = ('epoch: {} -- '+
                                   'training loss:{:.3f} -- '+
                                   'test loss: {:.3f}')
                print(training_string.format(i, training_loss, test_loss))
                with open('rnn_training_log.txt','a') as f:
                    f.write(training_string.format(i, training_loss, test_loss))
                    f.write('\n')

    def predict_proba(self, X):
        """predicts the probability of positive or negative
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, 'checkpoints/rnn.ckpt')
            feed_dict = {self.x_input:X}
            prob = sess.run(tf.nn.softmax(self.probs),
                            feed_dict=feed_dict)
        return prob


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
                 filter_sizes, num_filters, learning_rate=.01,
                 batch_size=100, num_epochs=1000,
                 dropout_keep_prob=.6, tokenization='word'):
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
        if tokenization=='word':
            self.create_word_dict()
            self.encode_tweets = self.encode_tweets_word2vec
        elif tokenization=='char':
            self.load_tweets_char_level()
            self.encode_tweets = self.encode_tweets_char_level
        else:
            raise Exception('Invalid option for tokenization'+
                            'valid options are char and word')
        self.setup_graph()

    def load_tweets_char_level(self):
        dataframe = pd.read_csv('../data/training.1600000.processed.noemoticon.csv',
                                encoding='iso8859', header=None,
                                names=['sentiment', 'id', 'time', 'query', 'user',
                                       'text'])
        labels = dataframe.sentiment/4
        tweets = dataframe.text
        tweets, labels = shuffle(tweets, labels)
        tweets = [tweet[:self.sequence_length] for tweet in tweets]
        lengths = [len(tweet) for tweet in tweets]

        tokenized_tweets = [tweet for tweet, length in zip(tweets, lengths)
                            if length > 0]
        labels = [label for label, length in zip(labels, lengths)
                  if length > 0]
        lengths = [length for length in lengths if length > 0]

        self.tweets = tweets
        self.labels = labels
        self.lengths = lengths
        return

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

    def encode_tweets_word2vec(self,tweets):
        tokenized_tweets = [[list(self.word_to_vec[word]) for word in tweet if
                             word in self.word_to_vec.vocab] for tweet in
                            tweets]
        pad = [0. for i in range(self.embedding_size)]
        tokenized_tweets = [[tweet[i] if i < len(tweet) else pad for i in
                             range(self.sequence_length)] for tweet in
                            tokenized_tweets]
        return tokenized_tweets

    def encode_tweets_char_level(self, tweets):
        vectorized_tweets = [[ord(tweet[i]) if i<len(tweet) else 0 for i
                              in range(self.sequence_length)] for tweet
                             in tweets]
        return vectorized_tweets

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
            loss = sess.run(self.test_loss,
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
                with open('training_log.txt','a') as f:
                    f.write(training_string.format(i, training_loss, test_loss))
                    f.write('\n')

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

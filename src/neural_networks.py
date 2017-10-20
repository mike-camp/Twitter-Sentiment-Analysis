"""A collection of neural network implementations
for classifying sentiment
"""
import tensorflow as tf
import numpy as np
from src import utils

class RNN(object):
    """A class for creating a character level
    recurrent neural network
    """

    def __init__(self, n_epochs, embedding_dim=126, batch_size=100,
                 cell_type='rnn', n_hidden=100, n_classes=2,
                learning_rate=.01, max_sequence_length=140,
                l2=.01):
        self.n_epochs = n_epochs
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.cell_type='rnn'
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.max_sequence_length = max_sequence_length
        self.lr = learning_rate
        self.l2=l2

    def split(self,ls, sublist_length=2000):
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
                range(0,len(ls),sublist_length)]

    def partial_fit(self, sess, tweets, labels):
        """partially fits a neural network to classify tweets

        Parameters:
        ----------
        tweets, list(str)
            list of tweets to classify
        labels, list(int)
            list of tweet labels
        """
        tweets, lengths = utils._encode_tweet_collection(tweets,
                                                   self.max_sequence_length,
                                                    self.embedding_dim)
        labels = np.array(labels)
        for batch_x, batch_length, batch_label in self._get_mini_batches(
                                                        tweets, lengths, labels):
            feed_dict = {self.x_input:batch_x,
                             self.length_input:batch_length,
                             self.label_input:batch_label}
            sess.run((self.optimizer, self.accuracy_update,
                      self.precision_update,
                     self.recall_update),feed_dict = feed_dict)

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
                                    on_value=1,off_value=0,dtype=tf.int32)
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
                    self.partial_fit(sess,tweets[split],labels[split])
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

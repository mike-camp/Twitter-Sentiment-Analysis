"""A collection of neural network implementations
for classifying sentiment
"""
import tensorflow as tf
import numpy as np

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

    def _encode_char(self, char):
        """encodes a character as a 1 hot vector of
        length 128.

        Parameters:
        -----------
        char: str
            character to encode

        Returns:
        --------
        nparray of length 128
        """
        if char == '':
            return np.zeros(self.embedding_dim)
        temp_array = np.zeros(self.embedding_dim)
        if ord(char) < self.embedding_dim:
            temp_array[ord(char)] = 1
        else:
            temp_array[0] = 1
        return temp_array

    def _encode_tweet(self, tweet):
        """Encodes a tweet as a series of character vectors
        and pads the tweet if it is shorter than 140 characters

        Parameters:
        -----------
        tweet: str
            the tweet to be encoded

        Returns:
        --------
        nparray of character vectors, shape = [140,128]
        """
        def get_char(i):
            return tweet[i] if i<len(tweet) else ''
        padded = np.vstack([self._encode_char(get_char(i)) for i in range(
            self.max_sequence_length)])
        return padded


    def _encode_tweet_collection(self, tweets):
        """takes a collection of tweets and converts them into
        an array of shape [n_tweets,n_characters,n_encoding]

        Parameters:
        -----------
        tweets: list(str)
            list of tweets

        Returns:
        --------
        encodings:
            np array of shape [n_tweets,n_characters,n_encoding]
        lengths:
            array of tweet lengths, shape=[n_tweets]
        """
        encodings = np.stack([self._encode_tweet(tweet) for tweet in tweets])
        lengths = [min(len(tweet),self.max_sequence_length) for tweet in tweets]
        return encodings, np.array(lengths)

    def fit(self, tweets, labels):
        """Fits a neural network to classify tweets

        Paramters:
        ----------
        tweets, list(str)
            list of tweets to classify
        labels, list(int)
            list of tweet labels
        """
        self._i = 0
        tweets, lengths = self._encode_tweet_collection(tweets)
        x_input,length_input,label_input = self._get_placeholders()
        print(len(tweets),len(lengths),len(labels))

        self._indices = np.arange(tweets.shape[0])
        labels = np.array(labels)

        logit_predictions = self._get_predictions(x_input, length_input)
        one_hot_labels = tf.one_hot(indices=label_input,depth=self.n_classes,
                                    on_value=1,off_value=0,dtype=tf.int32)
        cost, optimizer = self._optimize(logit_predictions, one_hot_labels)
        predictions = tf.argmax(logit_predictions, 1)

        with tf.name_scope('measurements'):
            precision, precision_update =\
                tf.contrib.metrics.streaming_precision(predictions,label_input)
            recall, recall_update = \
                tf.contrib.metrics.streaming_recall(predictions,label_input)
            accuracy, accuracy_update = \
                tf.contrib.metrics.streaming_accuracy(predictions, label_input)
 
        # Store variables related to metrics in a list which allows all 
        # measurement variables to be reset
        stream_vars = [i for i in tf.local_variables() if
                        i.name.split('/')[0] == 'measurements']
        reset_measurements = [tf.variables_initializer(stream_vars)]
 
        with tf.Session() as sess:
            print('running')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            #for epoch in range(self.n_epochs):
            for run in range(self.n_epochs*len(tweets)//self.batch_size):
                batch_x,batch_length,batch_label = self._get_mini_batch(tweets,
                                                                    lengths,
                                                                    labels)
                feed_dict = {x_input:batch_x, length_input:batch_length,
                             label_input:batch_label}
                loss,*_ = sess.run((cost, optimizer, precision_update,
                                    recall_update, accuracy_update),
                                   feed_dict=feed_dict)
                if self._i==1:
                    a, p, r = sess.run((accuracy, precision,recall),
                                   feed_dict=feed_dict)
                    sess.run(reset_measurements)
                    f1_score = 2*p*r/(p+r)
                    print("Testing Loss={:.4f}, testing accuracy={:.4f}, f1={:.5f}"\
                          .format(loss, a, f1_score))


    def predict(tweets):
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
        pass
    
    def score(tweets,predictions):
        pass


    def _get_placeholders(self):
        x = tf.placeholder(tf.float32, shape=[None,
                                                self.max_sequence_length,
                                                self.embedding_dim])
        l = tf.placeholder(tf.int32, shape=[None])
        y = tf.placeholder(tf.int32, shape=[None])
        return x, l, y

    def _get_mini_batch(self, X, lengths, labels):
        """Creates minibatches of the training data and labels
        If the process has gone through the data, then the data is
        shuffled and minibatches are returned from the shuffle

        """
        if self._i*self.batch_size > X.shape[0]:
            self._indices = np.random.permutation(self._indices)
            self._i = 0
        indices = self._indices[self._i*self.batch_size:
                                (self._i+1)*self.batch_size]
        self._i += 1
        return X[indices],\
                lengths[indices],\
                labels[indices]

    def _get_predictions(self, X, lengths, reuse_variables=False):
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
        with tf.variable_scope('RNN',reuse=reuse_variables):
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
        #outputs ahs shape 100,140,100
        output_list = tf.unstack(outputs, self.max_sequence_length, 1)
        #output_list has shape 140,100,100
        batch_range = tf.range(tf.shape(outputs)[0])
        indices = tf.stack([batch_range,lengths-1],axis=1)
        # tf.stack([...],axis=1) has shape 100,2
        # last_rnn_output = tf.gather_nd(outputs, tf.stack(
        #     [tf.range(batch_size), lengths-1], axis=1))
        ### THE ERROR IS ON THE NEXT LINE< TF.GATHER_ND CHANGED
        ### FROM TENSORFLOW 1.1 to TENSORFLOW 1.3
        last_rnn_output = tf.gather_nd(outputs,indices)

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
        with tf.variable_scope('RNN', reuse=True):
            cost += self.l2*tf.nn.l2_loss(tf.get_variable('W'))
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)\
                .minimize(cost)
        return cost, optimizer




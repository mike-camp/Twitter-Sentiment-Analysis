"""Collection of functions for preprocessing tweets
and analyzing sentiment.
"""
import re
import numpy as np
from bs4 import BeautifulSoup
import multiprocessing

def parrallel_map(function, iterable, n_jobs=8):
    pool = multiprocessing.pool.Pool()
    return pool.map(function,iterable)

def _HTMLEntitiesToUnicode(text):
    """Converts HTML entities to unicode.  For example '&amp;' becomes
     '&'."""
    soup = BeautifulSoup(text, 'html.parser')
    return soup.text


def _remove_emoticons(tweet):
    """finds all emoticons, removes them from the
    tweet, and then returns the tweet with emoticons
    removed as well as a list of emoticons

    Parameters:
    -----------
    tweet: str
        contents of a tweet

    Returns
    -------
    tweet_no_emoticons:
        string of tweet with emoticons removed
    emoticons:
        list of emoticons
    """
    emoticons_re = r'(?:[:;])(?:[-<])?(?:[()/\\|<>])'
    emoticons = re.findall(emoticons_re, tweet)
    tweet = re.sub(emoticons_re, '', tweet)
    return tweet.strip(), emoticons


def clean_text(tweet):
    """takes a tweet, removes all url's and replaces them
    with the keyword WEBLINK, removes special characters
    (i.e. letters, ampersands, etc) and returns the processed
    text

    Parameters:
    -----------
    tweet: str,
        text of a given tweet

    Returns:
    --------
    String corresponding to processed tweet
    """
    tweet = _HTMLEntitiesToUnicode(tweet)
    # remove web addresses
    tweet = re.sub(r'(?:(https?://)|(www\.))(?:\S+)?', 'WEBLINK', tweet)
    # replace usernames with AT_USER
    tweet = re.sub(r'@\w{1,15}', 'AT_USER', tweet)
    # remove hashtags
    tweet = re.sub(r'#(\S+)', r'\1', tweet)
    return tweet.strip()


def tokenize(tweet):
    """ tokenizes a tweet preserving
    emoticons

    Parameters:
    -----------
    tweet: str
        contents of a given tweet

    Returns:
    --------
    list of tokens
    """
    tweet = clean_text(tweet)
    tweet, emoticons = _remove_emoticons(tweet)
    words = re.findall(r"(?u)\b\w[\w']+\b", tweet)
    for i in emoticons:
        words.append(i)
    return words


def _encode_char(char,embedding_dim=126):
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
        return np.zeros(embedding_dim)
    temp_array = np.zeros(embedding_dim)
    if ord(char) < embedding_dim:
        temp_array[ord(char)] = 1
    else:
        temp_array[0] = 1
    return temp_array

def _encode_tweet(tweet, embedding_dim=126, max_sequence_length=140):
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
    padded = np.vstack([_encode_char(get_char(i)) for i in range(
        max_sequence_length)])
    return padded


def _encode_tweet_collection(tweets,max_sequence_length, embedding_dim):
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
    #encodings = np.stack([self._encode_tweet(tweet) for tweet in tweets])
    encodings = parrallel_map(_encode_tweet,tweets,n_jobs=3)

    lengths = [min(len(tweet),max_sequence_length) for tweet in tweets]
    return encodings, np.array(lengths)


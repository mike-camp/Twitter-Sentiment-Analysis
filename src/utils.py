"""Collection of functions for preprocessing tweets
and analyzing sentiment.
"""
import re
from bs4 import BeautifulSoup


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
        """ tokenizes a tweet

        Parameters:
        -----------
        tweet: str
            contents of a given tweet

        Returns:
        --------
        list of tokens
        """
        pass

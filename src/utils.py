"""Collection of functions for preprocessing tweets
and analyzing sentiment.
"""
import re
from bs4 import BeautifulSoup

def _HTMLEntitiesToUnicode(text):
    """Converts HTML entities to unicode.  For example '&amp;' becomes
     '&'."""
    soup = BeautifulSoup(text,'html.parser')
    return soup.text

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
    #extract emoticons
    emoticons_re = r'(?::|;)(?:-|<)?(?:\(|\||\)\\|/|<)'
    tweet = re.sub(r'(?:(https?://)|(www\.))(?:\S+)?', 'WEBLINK', tweet)
    tweet = re.sub(r'@\S+', 'AT_USER', tweet)
    tweet = re.sub(r'#(\S+)', r'\1', tweet)
    return tweet

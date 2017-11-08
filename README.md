## Twitter Sentiment Analysis
Analyzes trending topics by sentiment and location, and provides easy to understand visualizations of the results.

### Synopsis
This program allows you to analyze either preselected topics, or the latest trends on twitter.  It uses a sentiment analysis algorithm to determine the sentiment relating to a topic in real time, where the tweets are coming from, and how each part of the country feels about a given topic. The results of this are then shown on an interactive map.

![map of patriots](images/patriots_count.png)

### Web Application: twittersentimentanalysis.com

At twittersentimentanalysis.com, you can see analysis of how america felt about the patriots, roll tide, and the latest daily trends on twitter (updated twice daily).

![website](images/website.png)

### Data Source

The training data was taken from [Sanders Analytics](http://www.sananalytics.com/lab/twitter-sentiment/) which classified 1.6M tweets by emoticon.   The state gis files were found from the [US Census](http://www2.census.gov/geo/tiger/GENZ2016/shp/cb_2016_us_state_20m.zip).   The actual tweets were scraped using tweepy to access the twitter streaming API.  

The data was stored using MongoDB.


### Description
In the src directory there are four modules that help scrape twitter data
  - utils: text preprocessing, and tokenization module
  - neural\_networks: a collection of CNN and RNN neural networks for text classification
  - model: contains TweetPredictor class which predicts sentiment of tweets
  - twitter\_scraper: functions and classes to help scrape tweets about a topic from using tweepy.
  - visualization: contains functions that will automatically generate visualizations.   If you want to create the maps, you will only need to use this module.  

### Required Packages
The required packages can be found in requirements.txt.  They are
  - Folium: generates interactive html maps generated from python
  - Tweepy: twitter scraping API frontend
  - Sklearn
  - pandas
  - geopandas: pandas with geojson, arcgis, etc support
  - Tensorflow
  - re
  - BeautifulSoup
  

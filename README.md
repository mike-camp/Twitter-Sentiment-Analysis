## Twitter Sentiment Analysis

### Synopsis
An app that analyzes twitter data in real time, filtering by location and topic.  It then runs a sentiment analysis algorithm to determine the sentiment relating to a topic in real time.   The results of this are then shown on an interactive map.  The app can both look at a specific user defined topic, as well as finding trending topics from Twitter.    

This can allow you to see not only what is trending on Twitter in real time, but also how the country feels about the currently trending topics and how each individual state feels about the current topic.   The app is hosted on a website at twittersentimentanalysis.com, where it is updated daily. 

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
  

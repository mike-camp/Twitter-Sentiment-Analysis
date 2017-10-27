"""A module for visualization of tweets on maps
"""
import folium
import geopandas as gpd
import jinja2
from src import twitter_scraper
from src import process_tweets


def visualize_sentiment(df):
    """Given a dataframe with columns, state and sentiment,
    creates a visualization of the sentiment of the entire US.
    """
    df_grouped = df[['state', 'sentiment']].groupby('state').mean()
    gdf = gpd.read_file('data/cb_2016_us_state_20m.dbf')
    merged_df = gdf.merge(df_grouped, left_on='NAME', right_index=True)
    data_df = merged_df[['NAME', 'sentiment']]
    geo_str = merged_df[['NAME', 'geometry']].to_json()
    map1 = folium.Map(location=[+37, -100],
                      tiles='Cartodb Positron',
                      zoom_start=4)
    map1.choropleth(geo_data=geo_str,
                    data=data_df,
                    columns=['NAME', 'sentiment'],
                    fill_color='RdBu',
                    key_on='feature.properties.NAME',
                    legend_name='sentiment')
    return map1


def visualize_count(df):
    """Creates a visualization of total number of tweets of a topic
    across the entire US and returns the mean sentiment felt about the
    topic across the entire US

    Parameters:
    -----------
    df: pd.DataFrame
        dataframe containing all tweets.  Must contain the columns
          - state
          - sentiment

    Returns:
    --------
    map: Choropleth map of the US, where the color refers to the total
         number of tweets
    avg_sentiment: The average sentiment of a topic
    """
    avg_sentiment = df.sentiment.mean()
    df_grouped = df[['sentiment', 'state']].groupby(['state']).count()
    gdf = gpd.read_file('data/cb_2016_us_state_20m.dbf')
    merged_df = gdf.merge(df_grouped, how='left', left_on='NAME',
                          right_index=True)
    merged_df = merged_df.fillna(0)
    data_df = merged_df[['NAME', 'sentiment']].fillna(0)
    geo_str = merged_df[['NAME', 'geometry']].to_json()
    map1 = folium.Map(location=[+37, -100],
                      tiles='Cartodb Positron',
                      zoom_start=4)
    map1.choropleth(geo_data=geo_str,
                    data=data_df,
                    columns=['NAME', 'sentiment'],
                    fill_color='YlGn',
                    legend_name='number of tweets',
                    name='topic: sentiment = {:.2f}'.format(avg_sentiment),
                    key_on='feature.properties.NAME')
    return map1, avg_sentiment


def create_daily_topic_maps(n_hours):
    """Creates three maps for the top trending topics of the day"""
    table, _, daily_topics = twitter_scraper.stream_trends(n_trends=3,
                                                           n_hours=n_hours)
    tweet_processor = process_tweets.TweetProcessor(
        'models/stemmed_lr.pk')
    df_list = [tweet_processor.process_database(table, topics=[topic])
                                                for topic in daily_topics]
    jinja_params = {}
    for i, dataframe in enumerate(df_list):
        map_, avg_sentiment = visualize_count(dataframe)
        map_.save('website/maps/daily_topics_{}.html'.format(i+1))
        jinja_params['topic_{}'.format(i+1)] = daily_topics[i]
        jinja_params['topic_score_{}'.format(i+1)] = '{:.2f}'.format(
            avg_sentiment)
    with open('website/daily_trends_template.html', 'r') as f:
        template = jinja2.Template(f.read())
    with open('website/daily_trends.html', 'w') as f:
        f.write(template.render(**jinja_params))
    with open('website/index_template.html','r') as f:
        template = jinja2.Template(f.read())
    with open('website/index.html','w') as f:
        f.write(template.render(**jinja_params))


def create_topic_maps(topic_list, topic_name, n_hours=None):
    """Given a list of topics, creates maps according to the topic"""
    table = twitter_scraper.stream_topics(topic_list, topic_name,
                                          n_hours=n_hours)

    tweet_processor = process_tweets.TweetProcessor(
        'models/stemmed_lr.pk')
    df_list = [tweet_processor.process_database(table, topics=[topic])
               for topic in topic_list]
    for topic,dataframe in zip(topic_list, df_list):
        map_, avg_sentiment = visualize_count(dataframe)
        map_.save('website/maps/{}_count.html'.format(topic))
    with open('website/maps/sentiment_{}.txt'.format(topic), 'w') as f:
        f.write('{:.2f}'.format(avg_sentiment))

    map_ = visualize_sentiment(dataframe)
    map_.save('website/maps/{}_sentiment.html'.format(topic))


def create_maps_from_database(database, topic_list):
    """Given a database and a list of topics, scrapes the database,
    generates visualizations, and then saves the visualizations
    """
    pass

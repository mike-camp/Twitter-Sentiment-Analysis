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
    df_grouped = df[['sentiment', 'state']].groupby(['state']).mean()
    gdf = gpd.read_file('../data/state_geojson.json')
    merged_df = gdf.merge(df_grouped, left_on='name', right_index=True)
    data_df = merged_df.iloc[:, [0, 3]]
    geo_str = gpd.GeoDataFrame(merged_df.iloc[:, [0, 2]]).to_json()
    map1 = folium.Map(location=[+35, -100],
                      tiles='Cartodb Positron',
                      zoom_start=4)
    map1.choropleth(geo_data=geo_str,
                    data=data_df,
                    columns=['name', 'sentiment'],
                    legend_name='sentiment',
                    fill_color='RdBu',
                    key_on='feature.properties.name')
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
    gdf = gpd.read_file('../data/state_geojson.json')
    merged_df = gdf.merge(df_grouped, left_on='name', right_index=True)
    merged_df = merged_df.fillna(0)
    data_df = merged_df.iloc[:, [0, 3]]
    geo_str = gpd.GeoDataFrame(merged_df.iloc[:, [0, 2]]).to_json()
    map1 = folium.Map(location=[+35, -100],
                      tiles='Cartodb Positron',
                      zoom_start=4)
    map1.choropleth(geo_data=geo_str,
                    data=data_df,
                    columns=['name', 'sentiment'],
                    fill_color='YlGn',
                    legend_name='number of tweets',
                    name='topic: sentiment = {:.2f}'.format(avg_sentiment),
                    key_on='feature.properties.name')
    return map1, avg_sentiment


def create_daily_topic_maps(n_hours):
    """Creates three maps for the top trending topics of the day"""
    table, _, daily_topics = twitter_scraper.stream_trends(n_trends=3,
                                                           n_hours=n_hours)
    tweet_processor = process_tweets.TweetProcessor('models/tfidf_logistic_reg.pk')
    df_list = [tweet_processor.process_database(table, topics=[topic])
                                                for topic in daily_topics]
    jinja_params = {}
    for i, dataframe in enumerate(df_list):
        map_, avg_sentiment = visualize_count(dataframe)
        map_.save('maps/daily_topics_{}.html'.format(i))
        jinja_params['topic_{}'.format(i+1)] = daily_topics[i]
        jinja_params['topic_score_{}'.format(i+1)] = '{:.2f}'.format(
            avg_sentiment)
    with open('daily_trends_template.html','r') as f:
        template = jinja2.Template(f.read())
    with open('daily_trends.html','w') as f:
        f.write(template.render(**jinja_params))

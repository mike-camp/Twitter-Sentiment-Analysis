"""A module for visualization of tweets on maps
"""
import folium
import json
import geopandas as gpd
import numpy as np


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

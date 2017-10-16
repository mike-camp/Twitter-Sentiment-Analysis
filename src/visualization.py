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
    with open('../data/state_geojson.json') as f:
        geo_str = f.read()
    df_grouped = df[['sentiment','state']].groupby('state').mean()
    gdf = gpd.read_file('../data/state_geojson.json')
    merged_df = gdf.merge(df_grouped, left_on='name', right_index=True)
    data_df = merged_df.iloc[:, [0, 3]]
    geo_str = gpd.GeoDataFrame(merged_df.iloc[:, [0, 2]]).to_json()
    threshold_scale = np.linspace(0.0, 1.0, 6, dtype=float)\
            .tolist()
    map1 = folium.Map(location=[-15, -60],
                      tiles='Cartodb Positron',
                      zoom_start=2)
    map1.choropleth(geo_data=geo_str,
                    data=data_df,
                    columns=['name', 'sentiment'],
                    fill_color='RdBu',
                    key_on='feature.properties.name',
                    threshold_scale=threshold_scale)
    return map1


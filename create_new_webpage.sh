#!/bin/bash
/home/ubuntu/anaconda3/bin/python create_new_webpage.py
cd webpage
aws s3 sync . s3://mike-camp-twitter-sentiment-analysis
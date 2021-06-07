from flask import Flask, render_template, request
import string
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import math
import io
import os

from tweet_dataframe import *
from tweet_acquisition import *

#sentiment
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# 19-20 data
total_20_df = TweetData('resources/19-20-tweets*').build()
handle1_20_df = TweetData('resources/19-20-tweets/CrosstownLondon.jsonl').build()
handle2_20_df = TweetData('resources/19-20-tweets/GeesRestaurant.jsonl').build()
handle3_20_df = TweetData('resources/19-20-tweets/NMBCo.jsonl').build()
handle4_20_df = TweetData('resources/19-20-tweets/O_Spencer.jsonl').build()
handle5_20_df = TweetData('resources/19-20-tweets/The_pig_hotel.jsonl').build()

# # 20-21 data
total_21_df = TweetData('resources/20-21-tweets*').build()
handle1_21_df = TweetData('resources/20-21-tweets/CrosstownLondon.jsonl').build()
handle2_21_df = TweetData('resources/20-21-tweets/GeesRestaurant.jsonl').build()
handle3_21_df = TweetData('resources/20-21-tweets/NMBCo.jsonl').build()
handle4_21_df = TweetData('resources/20-21-tweets/O_Spencer.jsonl').build()
handle5_21_df = TweetData('resources/20-21-tweets/The_pig_hotel.jsonl').build()

final_total_20_df = set_df_name(total_20_df, 'Total 12-2019 - 02-2020')
final_handle1_20_df = set_df_name(handle1_20_df, 'Crosstown 12-2019 - 02-2020')
final_handle2_20_df = set_df_name(handle2_20_df, 'Gees Restaurant 12-2019 - 02-2020')
final_handle3_20_df = set_df_name(handle3_20_df, 'NMBCo 12-2019 - 02-2020')
final_handle4_20_df = set_df_name(handle4_20_df, 'Oliver Spencer 12-2019 - 02-2020')
final_handle5_20_df = set_df_name(handle5_20_df, 'The Pig Hotel 12-2019 - 02-2020')

final_total_21_df = set_df_name(total_21_df, 'Total 12-2020 - 02-2021')
final_handle1_21_df = set_df_name(handle1_21_df, 'Crosstown 12-2020 - 02-2021')
final_handle2_21_df = set_df_name(handle2_21_df, 'Gees Restaurant 12-2020 - 02-2021')
final_handle3_21_df = set_df_name(handle3_21_df, 'NMBCo 12-2020 - 02-2021')
final_handle4_21_df = set_df_name(handle4_21_df, 'Oliver Spencer 12-2020 - 02-2021')
final_handle5_21_df = set_df_name(handle5_21_df, 'The Pig Hotel 12-2020 - 02-2021')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/request-data', methods = ['POST', 'GET'])
def request_data():
    if request.method == 'POST':
        form_data = request.form
        return render_template('request-data.html', form_data = form_data)
    else:
        return '405 Method Not Allowed'



@app.route('/likes')
def likes_page():
    
    # displays likes statisics
    def show_likes_stats(df):
        drop_retweets(df)

        likes_sum = df["favorite_count"].sum()
        likes_mean = round(df["favorite_count"].mean(),2)
        file_path = f'static\images\likes\{df.__str__} Likes Countplot.png'
     
        df_countplot = sns.countplot(data=df, x='favorite_count', color='blue')
        #df_countplot.set_ticklabels(df_countplot.get_ticklabels())

        #print()
        if len(df_countplot.get_xticks()) > 20:
            df_countplot.set_xticklabels(df_countplot.get_xticks(), rotation=90)
            for label in df_countplot.set_xticks(df_countplot.get_xticks())[::2]:
                label.set_visible(False)
        
        plt.xlabel("No. of likes")
        plt.ylabel("Tweets")

        df_countplot.set_title(f'{df.__str__}')
        
        if os.path.isfile(file_path):
            pass
        else:
            plt.savefig(file_path)
            plt.close()

        return likes_sum, likes_mean, file_path
        
    # Shows all likes stats
    total_likes_sum20, total_likes_mean20, total_likes_cp20 = show_likes_stats(final_total_20_df)
    handle1_likes_sum20, handle1_likes_mean20, handle1_likes_cp20 = show_likes_stats(final_handle1_20_df)
    handle2_likes_sum20, handle2_likes_mean20, handle2_likes_cp20 = show_likes_stats(final_handle2_20_df)
    handle3_likes_sum20, handle3_likes_mean20, handle3_likes_cp20 = show_likes_stats(final_handle3_20_df)
    handle4_likes_sum20, handle4_likes_mean20, handle4_likes_cp20 = show_likes_stats(final_handle4_20_df)
    handle5_likes_sum20, handle5_likes_mean20, handle5_likes_cp20 = show_likes_stats(final_handle5_20_df)

    total_likes_sum21, total_likes_mean21, total_likes_cp21 = show_likes_stats(final_total_21_df)
    handle1_likes_sum21, handle1_likes_mean21, handle1_likes_cp21 = show_likes_stats(final_handle1_21_df)
    handle2_likes_sum21, handle2_likes_mean21, handle2_likes_cp21 = show_likes_stats(final_handle2_21_df)
    handle3_likes_sum21, handle3_likes_mean21, handle3_likes_cp21 = show_likes_stats(final_handle3_21_df)
    handle4_likes_sum21, handle4_likes_mean21, handle4_likes_cp21 = show_likes_stats(final_handle4_21_df)
    handle5_likes_sum21, handle5_likes_mean21, handle5_likes_cp21 = show_likes_stats(final_handle5_21_df)


    # set difference in stats variables
    total_likes_sum_diff = set_difference(total_likes_sum20, total_likes_sum21)
    handle1_likes_sum_diff = set_difference(handle1_likes_sum20, handle1_likes_sum21)
    handle2_likes_sum_diff = set_difference(handle2_likes_sum20, handle2_likes_sum21)
    handle3_likes_sum_diff = set_difference(handle3_likes_sum20, handle3_likes_sum21)
    handle4_likes_sum_diff = set_difference(handle4_likes_sum20, handle4_likes_sum21)
    handle5_likes_sum_diff = set_difference(handle5_likes_sum20, handle5_likes_sum21)

    total_likes_mean_diff = set_difference(total_likes_mean20, total_likes_mean21)
    handle1_likes_mean_diff = set_difference(handle1_likes_mean20, handle1_likes_mean21)
    handle2_likes_mean_diff = set_difference(handle2_likes_mean20, handle2_likes_mean21)
    handle3_likes_mean_diff = set_difference(handle3_likes_mean20, handle3_likes_mean21)
    handle4_likes_mean_diff = set_difference(handle4_likes_mean20, handle4_likes_mean21)
    handle5_likes_mean_diff = set_difference(handle5_likes_mean20, handle5_likes_mean21)

    return render_template('likes.html', **locals())

@app.route('/retweets')
def retweet_page():
    def show_retweets_stats(df):
        drop_retweets(df)

        retweets_sum = df["retweet_count"].sum()
        retweets_mean = round(df["retweet_count"].mean(),2)
        file_path = f'static\images\\retweets\{df.__str__} Retweets Countplot.png'
        
        df_countplot = sns.countplot(data=df, x='retweet_count', color='green')

        #print()
        if len(df_countplot.get_xticks()) > 20:
            df_countplot.set_xticklabels(df_countplot.get_xticks(), rotation=90)
            for label in df_countplot.set_xticks(df_countplot.get_xticks())[::2]:
                label.set_visible(False)
        
        plt.xlabel("No. of retweets")
        plt.ylabel("Tweets")

        df_countplot.set_title(f'{df.__str__}')
        
        if os.path.isfile(file_path):
            pass
        else:
            plt.savefig(file_path)
            plt.close()

        return retweets_sum, retweets_mean, file_path

    # Shows all retweets stats
    total_retweets_sum20, total_retweets_mean20, total_retweets_cp20 = show_retweets_stats(final_total_20_df)
    handle1_retweets_sum20, handle1_retweets_mean20, handle1_retweets_cp20 = show_retweets_stats(final_handle1_20_df)
    handle2_retweets_sum20, handle2_retweets_mean20, handle2_retweets_cp20 = show_retweets_stats(final_handle2_20_df)
    handle3_retweets_sum20, handle3_retweets_mean20, handle3_retweets_cp20 = show_retweets_stats(final_handle3_20_df)
    handle4_retweets_sum20, handle4_retweets_mean20, handle4_retweets_cp20 = show_retweets_stats(final_handle4_20_df)
    handle5_retweets_sum20, handle5_retweets_mean20, handle5_retweets_cp20 = show_retweets_stats(final_handle5_20_df)

    total_retweets_sum21, total_retweets_mean21, total_retweets_cp21 = show_retweets_stats(final_total_21_df)
    handle1_retweets_sum21, handle1_retweets_mean21, handle1_retweets_cp21 = show_retweets_stats(final_handle1_21_df)
    handle2_retweets_sum21, handle2_retweets_mean21, handle2_retweets_cp21 = show_retweets_stats(final_handle2_21_df)
    handle3_retweets_sum21, handle3_retweets_mean21, handle3_retweets_cp21 = show_retweets_stats(final_handle3_21_df)
    handle4_retweets_sum21, handle4_retweets_mean21, handle4_retweets_cp21 = show_retweets_stats(final_handle4_21_df)
    handle5_retweets_sum21, handle5_retweets_mean21, handle5_retweets_cp21 = show_retweets_stats(final_handle5_21_df)

    total_retweets_sum_diff = set_difference(total_retweets_sum20, total_retweets_sum21)
    handle1_retweets_sum_diff = set_difference(handle1_retweets_sum20, handle1_retweets_sum21)
    handle2_retweets_sum_diff = set_difference(handle2_retweets_sum20, handle2_retweets_sum21)
    handle3_retweets_sum_diff = set_difference(handle3_retweets_sum20, handle3_retweets_sum21)
    handle4_retweets_sum_diff = set_difference(handle4_retweets_sum20, handle4_retweets_sum21)
    handle5_retweets_sum_diff = set_difference(handle5_retweets_sum20, handle5_retweets_sum21)

    total_retweets_mean_diff = set_difference(total_retweets_mean20, total_retweets_mean21)
    handle1_retweets_mean_diff = set_difference(handle1_retweets_mean20, handle1_retweets_mean21)
    handle2_retweets_mean_diff = set_difference(handle2_retweets_mean20, handle2_retweets_mean21)
    handle3_retweets_mean_diff = set_difference(handle3_retweets_mean20, handle3_retweets_mean21)
    handle4_retweets_mean_diff = set_difference(handle4_retweets_mean20, handle4_retweets_mean21)
    handle5_retweets_mean_diff = set_difference(handle5_retweets_mean20, handle5_retweets_mean21)

    return render_template('retweets.html', **locals())

@app.route('/replies')
def replies_page():
    def show_replies_stats(df):
        drop_retweets(df)

        replies_sum = df["reply_count"].sum()
        replies_mean = round(df["reply_count"].mean(),2)
        file_path = f'static\images\\replies\{df.__str__} Replies Countplot.png'
        
        df_countplot = sns.countplot(data=df, x='reply_count', color='yellow')

        if len(df_countplot.get_xticks()) > 20:
            df_countplot.set_xticklabels(df_countplot.get_xticks(), rotation=90)
            for label in df_countplot.set_xticks(df_countplot.get_xticks())[::2]:
                label.set_visible(False)
        
        plt.xlabel("No. of replies")
        plt.ylabel("Tweets")

        df_countplot.set_title(f'{df.__str__}')
        
        if os.path.isfile(file_path):
            pass
        else:
            plt.savefig(file_path)
            plt.close()

        return replies_sum, replies_mean, file_path
    
    # # print('Replies 12-2019 - 02-2020\n')
    total_replies_sum20, total_replies_mean20, total_replies_cp20 = show_replies_stats(final_total_20_df)
    handle1_replies_sum20, handle1_replies_mean20, handle1_replies_cp20 = show_replies_stats(final_handle1_20_df)
    handle2_replies_sum20, handle2_replies_mean20, handle2_replies_cp20 = show_replies_stats(final_handle2_20_df)
    handle3_replies_sum20, handle3_replies_mean20, handle3_replies_cp20 = show_replies_stats(final_handle3_20_df)
    handle4_replies_sum20, handle4_replies_mean20, handle4_replies_cp20 = show_replies_stats(final_handle4_20_df)
    handle5_replies_sum20, handle5_replies_mean20, handle5_replies_cp20 = show_replies_stats(final_handle5_20_df)

    total_replies_sum21, total_replies_mean21, total_replies_cp21 = show_replies_stats(final_total_21_df)
    handle1_replies_sum21, handle1_replies_mean21, handle1_replies_cp21 = show_replies_stats(final_handle1_21_df)
    handle2_replies_sum21, handle2_replies_mean21, handle2_replies_cp21 = show_replies_stats(final_handle2_21_df)
    handle3_replies_sum21, handle3_replies_mean21, handle3_replies_cp21 = show_replies_stats(final_handle3_21_df)
    handle4_replies_sum21, handle4_replies_mean21, handle4_replies_cp21 = show_replies_stats(final_handle4_21_df)
    handle5_replies_sum21, handle5_replies_mean21, handle5_replies_cp21 = show_replies_stats(final_handle5_21_df)

    # set difference in stats variables
    total_replies_sum_diff = set_difference(total_replies_sum20, total_replies_sum21)
    handle1_replies_sum_diff = set_difference(handle1_replies_sum20, handle1_replies_sum21)
    handle2_replies_sum_diff = set_difference(handle2_replies_sum20, handle2_replies_sum21)
    handle3_replies_sum_diff = set_difference(handle3_replies_sum20, handle3_replies_sum21)
    handle4_replies_sum_diff = set_difference(handle4_replies_sum20, handle4_replies_sum21)
    handle5_replies_sum_diff = set_difference(handle5_replies_sum20, handle5_replies_sum21)

    total_replies_mean_diff = set_difference(total_replies_mean20, total_replies_mean21)
    handle1_replies_mean_diff = set_difference(handle1_replies_mean20, handle1_replies_mean21)
    handle2_replies_mean_diff = set_difference(handle2_replies_mean20, handle2_replies_mean21)
    handle3_replies_mean_diff = set_difference(handle3_replies_mean20, handle3_replies_mean21)
    handle4_replies_mean_diff = set_difference(handle4_replies_mean20, handle4_replies_mean21)
    handle5_replies_mean_diff = set_difference(handle5_replies_mean20, handle5_replies_mean21)

    return render_template('replies.html', **locals())


@app.route('/sentiment')
def sentiment_page():
    # removes punctuation and stopwords
    def clean_tweet(message):
        no_punc = [char for char in message if char not in string.punctuation]
        no_punc_joined = ''.join(no_punc)
        clean_string = [word for word in no_punc_joined.split() if word.lower() not in stopwords.words('english')]

        return clean_string

    def get_sentiment(df, txt_col):
        return df[txt_col].map(lambda txt: TextBlob(txt).sentiment.polarity)

    def set_sentiment(df, df_name):
        df_clean = df['text'].apply(clean_tweet)

        df_clean['text'] = df['text'].apply(str)

        #new column to show sentiment polarity
        df_clean['label'] = get_sentiment(df_clean, 'text')
        pol_df = df_clean['label'][df_clean['label'] != 0]

        final_pol_df = set_df_name(pol_df, df_name)
        
        return final_pol_df
    # create histplot with appropriate title
    def show_sentiment(df):
        polarity_mean = round(df.mean(),2)

        file_path = f'static\images\sentiment\{df.__str__} sentiment histplot.png'
        df_hist = df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1], color='purple')
        df_hist.set_title(f'{df.__str__}')

        if os.path.isfile(file_path):
            pass
        else:
            plt.savefig(file_path)
            plt.close()
        return file_path, polarity_mean

    # define dataframes
    total_20_df = TweetData('resources/19-20-mentions*').build()
    handle2_20_df = TweetData('resources/19-20-mentions/GeesRestaurant.jsonl').build()
    handle3_20_df = TweetData('resources/19-20-mentions/NMBCo.jsonl').build()
    handle4_20_df = TweetData('resources/19-20-mentions/O_Spencer.jsonl').build()
    handle5_20_df = TweetData('resources/19-20-mentions/The_pig_hotel.jsonl').build()

    # # 20-21 data
    total_21_df = TweetData('resources/20-21-mentions*').build()
    handle2_21_df = TweetData('resources/20-21-mentions/GeesRestaurant.jsonl').build()
    handle3_21_df = TweetData('resources/20-21-mentions/NMBCo.jsonl').build()
    handle4_21_df = TweetData('resources/20-21-mentions/O_Spencer.jsonl').build()
    handle5_21_df = TweetData('resources/20-21-mentions/The_pig_hotel.jsonl').build()


    pol_total_20_df = set_sentiment(total_20_df, 'Total 12-2019 - 02-2020 Sentiment Polarity')
    pol_handle2_20_df = set_sentiment(handle2_20_df, 'Gees 12-2019 - 02-2020 Sentiment Polarity')
    pol_handle3_20_df = set_sentiment(handle3_20_df, 'NMBCo 12-2019 - 02-2020 Sentiment Polarity')
    pol_handle4_20_df = set_sentiment(handle4_20_df, 'Oliver Spencer 12-2019 - 02-2020 Sentiment Polarity')
    pol_handle5_20_df = set_sentiment(handle5_20_df, 'The Pig Hotel 12-2019 - 02-2020 Sentiment Polarity')


    pol_total_21_df = set_sentiment(handle1_21_df, 'Total 12-2020 - 02-2021 Sentiment Polarity')
    pol_handle2_21_df = set_sentiment(handle2_21_df, 'Gees 12-2020 - 02-2021 Sentiment Polarity')
    pol_handle3_21_df = set_sentiment(handle3_21_df, 'NMBCo 12-2020 - 02-2021 Sentiment Polarity')
    pol_handle4_21_df = set_sentiment(handle4_21_df, 'Oliver Spencer 12-2020 - 02-2021 Sentiment Polarity')
    pol_handle5_21_df = set_sentiment(handle5_21_df, 'The Pig Hotel 12-2020 - 02-2021 Sentiment Polarity')


    total_sent_pol_20, total_mean_pol_20 = show_sentiment(pol_total_20_df)
    handle2_sent_pol_20, handle2_mean_pol_20 = show_sentiment(pol_handle2_20_df)
    handle3_sent_pol_20, handle3_mean_pol_20 = show_sentiment(pol_handle3_20_df)
    handle4_sent_pol_20, handle4_mean_pol_20 = show_sentiment(pol_handle4_20_df)
    handle5_sent_pol_20, handle5_mean_pol_20 = show_sentiment(pol_handle5_20_df)

    total_sent_pol_21, total_mean_pol_21 = show_sentiment(pol_total_21_df)
    handle2_sent_pol_21, handle2_mean_pol_21 = show_sentiment(pol_handle2_21_df)
    handle3_sent_pol_21, handle3_mean_pol_21 = show_sentiment(pol_handle3_21_df)
    handle4_sent_pol_21, handle4_mean_pol_21 = show_sentiment(pol_handle4_21_df)
    handle5_sent_pol_21, handle5_mean_pol_21 = show_sentiment(pol_handle5_21_df)

    total_sent_pol_mean_diff = set_difference(total_mean_pol_20, total_mean_pol_21)
    handle2_sent_pol_mean_diff = set_difference(handle2_mean_pol_20, handle2_mean_pol_21)
    handle3_sent_pol_mean_diff = set_difference(handle3_mean_pol_20, handle3_mean_pol_21)
    handle4_sent_pol_mean_diff = set_difference(handle4_mean_pol_20, handle4_mean_pol_21)
    handle5_sent_pol_mean_diff = set_difference(handle5_mean_pol_20, handle5_mean_pol_21)


    return render_template('sentiment.html', **locals())


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
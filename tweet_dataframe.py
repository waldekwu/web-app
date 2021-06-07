import jsonlines
import glob
import os, json
import pandas as pd

#this class allows to quickly build multiple dataframes using only file location
class TweetData:
    def __init__(self, file_path):
        self.file_path = file_path
    def build(self):
        if self.file_path[-1] == '*':
            json_pattern = os.path.join(self.file_path,'*.jsonl')
            tweet_list = glob.glob(json_pattern)
            data_frames = []

        # read data and append to the list
            for f in tweet_list:
                data = pd.read_json(f, lines=True)
                data_frames.append(data)
        # concat all data frames in the list
            df = pd.concat(data_frames, ignore_index=True)
        else:
            with jsonlines.open(self.file_path) as f:
                df = pd.DataFrame(f)
        return df
    
def set_df_name(df, file_name):
    df.__str__ = file_name
    return df

# Removes retweets from a dataframe and assigns a custom __str__
def drop_retweets(df):
    if 'retweeted_status' in df:
        df['is_retweet'] = df['retweeted_status']
        # replaces NaN
        df['is_retweet'].fillna('no', inplace=True)
        # replaces RT
        for i in df['is_retweet']:
            if type(i) is dict:
                df['is_retweet'] = df['is_retweet'].replace([i], 'yes')
        # removes RT
        df = df[df.is_retweet != 'yes']
    return df

def set_difference(data_20, data_21):
    total = data_21 - data_20
    total_rounded = round(total, 2)
    if total_rounded > 0:
        total_rounded = '+' + str(total_rounded)
    else:
        pass
    return total_rounded
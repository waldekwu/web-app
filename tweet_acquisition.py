import yaml
import json
from searchtweets import load_credentials, gen_rule_payload, ResultStream

# connect to Full Archieve API
def authenticate():
    config = dict(
        search_tweets_api = dict(
            account_type = 'premium',
            endpoint = 'https://api.twitter.com/1.1/tweets/search/fullarchive/dev.json',
            consumer_key = '7hTX1E00po4lupuP9940SCVl3',
            consumer_secret = 'ofMc9limib4WWnZVANnTx3nHvR8XgYGWlO1XBAh0iX8cL4oXfw'
        )
    )

    with open('twitter_keys_fullarchive.yaml', 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    premium_search_args = load_credentials("twitter_keys_fullarchive.yaml",
                                        yaml_key="search_tweets_api",
                                        env_overwrite=False)
    return premium_search_args

def get_handle_tweets(handle, from_date, to_date):

    premium_search_args = authenticate()

    FILENAME = f'resources\handle\tweets\{handle + to_date}.jsonl'  # Where the Tweets should be saved
    PRINT_AFTER_X = 10

    rule = gen_rule_payload(f'from:{handle}', 
                            from_date = from_date,
                            to_date = to_date)
                            
    rs = ResultStream(rule_payload=rule,
                    **premium_search_args)

    with open(FILENAME, 'a', encoding='utf-8') as f:
        n = 0
        for tweet in rs.stream():
            n += 1
            if n % PRINT_AFTER_X == 0:
                print('{0}: {1}'.format(str(n), tweet['created_at']))
            json.dump(tweet, f)
            f.write('\n')
    return 'Success'

def get_handle_mentions_tweets(handle, from_date, to_date):

    premium_search_args = authenticate()

    FILENAME = f'resources\handle\mentions-tweets\{handle + to_date}_mentions.jsonl'  # Where the Tweets should be saved
    PRINT_AFTER_X = 10

    rule = gen_rule_payload(f'@{handle}', 
                            from_date = from_date,
                            to_date = to_date)
                            
    rs = ResultStream(rule_payload=rule,
                        max_results=300,
                    **premium_search_args)

    with open(FILENAME, 'a', encoding='utf-8') as f:
        n = 0
        for tweet in rs.stream():
            n += 1
            if n % PRINT_AFTER_X == 0:
                print('{0}: {1}'.format(str(n), tweet['created_at']))
            json.dump(tweet, f)
            f.write('\n')
    return 'Success'
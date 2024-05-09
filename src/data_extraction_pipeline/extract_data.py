import os, json, sys
import argparse
import pandas as pd
from datetime import date, timedelta
import requests

from collections import defaultdict
import pickle

sys.path.append('..')
from api_keys import *
from utils import NewsArticleDataset


# GLOBALS

# How many days before our chosen date do we fetch news articles from
NUM_DAYS = 2
TICKER_TO_DOWNLOAD = {"AAPL", "AMZN", "NVDA", "F", "VZ", "AAPL", "TSLA", "BA", "BAC", "ILMN", "MMM"}

def augment_forex_data(dataset, acquired_data_dir):
    # read forex.csv
    forex_df = pd.read_csv(acquired_data_dir + '/forex/forex.csv')
    # keep only the columns title, text, true_sentiment
    forex_df = forex_df[['title', 'text', 'true_sentiment']]
    # concatenate title and text into a new column called content
    forex_df['content'] = forex_df['title'] + ' ' + forex_df['text']
    # lower case the true_sentiment column
    forex_df['true_sentiment'] = forex_df['true_sentiment'].str.lower()
    data = forex_df['content'].tolist()
    labels = forex_df['true_sentiment'].tolist()

    dataset.dataset['data'] += data
    dataset.dataset['labels'] += labels


'''
Pipeline for extracting acquired data into separate documents 

'''

def main():
    # parse any commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquired_data_dir", type=str, default='../../data/raw_data', help="Path to the directory containing acquired data.")
    parser.add_argument("--output_dir", type=str, default='../../data/extracted_data', help="Path to the directory where acquired data should be stored.")
    parser.add_argument("--debug", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("--write_cache", default=False, help="Setting flag to true writes a cache to load dataset with.", action="store_true")
    #Add any more arguments as and when needed
    args = parser.parse_args()


    '''
    Input Directory Structure:
                        base_dir -> ticker_name -> from-date_to-date.json

    Output Directory Structure:
                       output_dir -> ticker_name -> to-date -> content_date.txt
    '''
    if args.debug:
        print(f"Extracting data from {args.acquired_data_dir}")
    # Read json that contains tickers and dates being considered
    if not os.path.isdir(args.acquired_data_dir):
        print(f"Directory {args.acquired_data_dir} doesn't exist.")
        exit(1)

    tickers = [d for d in os.listdir(args.acquired_data_dir) \
                   if os.path.isdir(os.path.join(args.acquired_data_dir,d)) and d != 'forex']
    ticker_news = defaultdict(set)
    list_articles = []

    dataset = NewsArticleDataset()

    for t in tickers:
        if args.debug:
            print(f"Extracting data for {t}")

        if t not in ticker_news.keys():
            ticker_news[t] = {}
        base_dir = os.path.join(args.acquired_data_dir,t)
        news_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f)) and \
                      f.split('_')[2].split('.')[0] == 'news']
        
        for f in news_files:
            date = f.split('_')[1]

            if args.debug:
                print(f"Extracting data for date {date}")
            with open(os.path.join(base_dir, f)) as file:
                news_data = json.loads(file.read())
                
                # ignore files we couldn't fetch for now
                if isinstance(news_data, dict) and "Note" in news_data:
                    continue

                dataset.add_to_dataset(news_data, t.upper())
    
    augment_forex_data(dataset, args.acquired_data_dir)

    dataset.write_dataset(args.output_dir)
    #Cache
    if args.write_cache:
        dataset.write_cache(args.output_dir)

    return

if __name__ == '__main__':
    main()
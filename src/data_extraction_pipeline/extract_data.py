import os, json
import argparse
from datetime import date, timedelta
import requests

from collections import defaultdict
import pickle
from api_keys import *

from newsarticle import NewsArticle


# GLOBALS

# How many days before our chosen date do we fetch news articles from
NUM_DAYS = 2
TICKER_TO_DOWNLOAD = {"AAPL", "AMZN", "NVDA", "F", "VZ", "AAPL", "TSLA", "BA", "BAC", "ILMN", "MMM"}

'''
Pipeline for extracting acquired data into separate documents 

'''

def extract_news_data(news_data_dict, price_data):
    article = NewsArticle(news_data_dict["title"], \
                          news_data_dict["date"].split("T")[0], \
                          news_data_dict["content"])
    for price in price_data:
        if price["date"] == article.date:
            article.set_open_price(price["open"])
            article.set_close_price(price["adjusted_close"])
    return article


def main():
    # parse any commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquired_data_dir", type=str, default='../../data/raw_data', help="Path to the directory containing acquired data.")
    parser.add_argument("--output_dir", type=str, default='../../data/extracted_data', help="Path to the directory where acquired data should be stored.")
    parser.add_argument("--debug", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
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
               if os.path.isdir(os.path.join(args.acquired_data_dir,d))]
    ticker_news = defaultdict(set)
    list_articles = []

    for t in tickers:
        if args.debug:
            print(f"Extracting data for {t}")
        if t not in ticker_news.keys():
            ticker_news[t] = {}
        ticker_dates = set()
        base_dir = os.path.join(args.acquired_data_dir,t)
        news_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f)) and \
                      f.split('_')[2].split('.')[0] == 'news']
        for f in news_files:
            date = f.split('_')[1]

            if args.debug:
                print(f"Extracting data for date {date}")
            ticker_dates.add(date)
            with open(os.path.join(base_dir, f)) as file:
                news_data = json.loads(file.read())
                price_filename = None
                for fn in  os.listdir(base_dir):
                    if fn.split('_')[1] == date and fn.split('_')[2].split('.')[0] == 'price':
                        price_filename = fn
                        break
                if price_filename is None:
                    print(f'Couldn\'t find price file for date: {date}')
                    exit(1)

                with open(os.path.join(base_dir, f)) as price_file:
                    price_data = json.loads(price_file.read())
                    
                    if isinstance(news_data, list):
                        for news_article in news_data:
                            list_articles.append(extract_news_data(news_article, price_data))
                    else:
                        list_articles.appen(extract_news_data(news_data, price_data))

    with open(os.path.join(args.output_dir, 'extracted_articles.pkl'), 'wb') as f:
        pickle.dump(list_articles, f)

    return

if __name__ == '__main__':
    main()
import os, json
import argparse
from datetime import date, timedelta
import requests

import sys
sys.path.append('..')

from api_keys import *
# GLOBALS

# How many days before our chosen date do we fetch news articles from
NUM_DAYS_NEWS = 2
NUM_DAYS_PRICE = 1
TICKER_TO_DOWNLOAD = {"TSLA", "BA", "BAC", "ILMN", "MMM"}


def build_alphavantage_datetime(date: str):
    return ''.join(date.split('-'))

def build_alphavantage_url(s, from_date, to_date, limit=50, offset=0):
    return f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={s}&apikey={ALPHA_VANTAGE_API_KEY}&time_from={build_alphavantage_datetime(from_date)}T0000&time_to={build_alphavantage_datetime(to_date)}T2359'


def build_eodhd_url(s, from_date, to_date, limit=50, offset=0):
    return "https://eodhd.com/api/news?s=" + s +".US&offset="+offset+"&limit="+limit+"&from="+from_date+"&to="+to_date+"&api_token="+EODHD_API_KEY+"&fmt=json"

def fetch_news_data_for_ticker(ticker, from_date, to_date):
    url = build_alphavantage_url(ticker.lower(), from_date.isoformat(), to_date.isoformat(), 60, "0")
    return requests.get(url).json()

def fetch_price_data_for_ticker(ticker, from_date, to_date):
    url = f"https://eodhd.com/api/eod/{ticker}.US?api_token={EODHD_API_KEY}&fmt=json&from={from_date.isoformat()}&to={to_date.isoformat()}"
    return requests.get(url).json()

def fetch_news(args, ticker_dates, ticker):
    for date_str in ticker_dates[ticker]["dates"]:
        #If news data for this ticker and date combination was already saved, skip ahead!
        if not os.path.isdir(args.output_dir+os.sep+ticker.lower()):
            os.mkdir(args.output_dir+os.sep+ticker.lower())

        to_date = date.fromisoformat(date_str)
        from_date = to_date - timedelta(days=NUM_DAYS_NEWS)
        if args.debug:
            print(f"From Date: {build_alphavantage_datetime(from_date.isoformat())}")
            print(f"To Date: {build_alphavantage_datetime(to_date.isoformat())}")
        else:
            data = fetch_news_data_for_ticker(ticker, from_date, to_date)
            output_filename = args.output_dir+os.sep+ticker+os.sep+from_date.isoformat()+"_"+to_date.isoformat()+"_news.json"
            with open(output_filename, "w+") as f:
                json.dump(data, f)

def fetch_price(args, ticker_dates, ticker):
    for dateStr in ticker_dates[ticker]["dates"]:
        if not os.path.isdir(args.output_dir+os.sep+ticker.lower()):
            os.mkdir(args.output_dir+os.sep+ticker.lower())

        to_date = date.fromisoformat(dateStr)
        from_date = to_date - timedelta(days=NUM_DAYS_PRICE)
        if args.debug:
            print(f"From Date: {from_date.isoformat()}")
            print(f"To Date: {to_date.isoformat()}")
        else:
            data = fetch_price_data_for_ticker(ticker, from_date, to_date)
            output_filename = args.output_dir+os.sep+ticker+os.sep+from_date.isoformat()+"_"+to_date.isoformat()+"_price.json"
            with open(output_filename, "w+") as f:
                json.dump(data, f)
    

def main():
    # parse any commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker_file", type=str, default='ticker_data.json', help="Path to the JSON file that contains ticker names and dates.")
    parser.add_argument("--output_dir", type=str, default='../../data/raw_data', help="Path to the directory where acquired data should be stored.")
    parser.add_argument("--debug", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("--price", default=False, help="Setting flag to true fetches price data.", action="store_true")
    parser.add_argument("--news", default=False, help="Setting flag to true fetches news data.", action="store_true")

    #Add any more arguments as and when needed
    args = parser.parse_args()

    # Read json that contains tickers and dates being considered
    ticker_dates = None
    with open(args.ticker_file, "r") as f:
        ticker_dates = json.loads(f.read())
    
    if ticker_dates is None:
        print(f"Failed to read json file from path {args.ticker_file}")
        exit(1)

    for ticker in ticker_dates.keys():
        if ticker not in TICKER_TO_DOWNLOAD:
            continue
        if args.news:
            print(f"Fetching news for {ticker}")
            fetch_news(args, ticker_dates, ticker)
        if args.price:
            print(f"Fetching price for {ticker}")
            fetch_price(args, ticker_dates, ticker)

    return

if __name__ == '__main__':
    main()
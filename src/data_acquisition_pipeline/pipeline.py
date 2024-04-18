import os, json
import argparse
from datetime import date, timedelta
import requests

# API KEYS (If necessary)
from api_keys import EODHD_API_KEY

# GLOBALS

# How many days before our chosen date do we fetch news articles from
NUM_DAYS = 2


def build_eodhd_url(s, from_date, to_date, limit=50, offset=0):
    return "https://eodhd.com/api/news?s=" + s +".US&offset="+offset+"&limit="+limit+"&from="+from_date+"&to="+to_date+"&api_token="+EODHD_API_KEY+"&fmt=json"

def main():
    # parse any commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker_file", type=str, default='ticker_data.json', help="Path to the JSON file that contains ticker names and dates.")
    parser.add_argument("--output_dir", type=str, default='../../data', help="Path to the directory where acquired data should be stored.")
    #Add any more arguments as and when needed
    args = parser.parse_args()

    # Read json that contains tickers and dates being considered
    ticker_dates = None
    with open(args.ticker_file, "r") as f:
        ticker_dates = json.loads(f.read())
    
    if ticker_dates is None:
        print(f"Failed to read json file from path {args.ticker_file}")
        exit(1)
    
    news_data = {}

    for ticker in ticker_dates.keys():
        for date_str in ticker_dates[ticker]["dates"]:
            #If news data for this ticker and date combination was already saved, skip ahead!
            if not os.path.isdir(args.output_dir+os.sep+ticker.lower()):
                os.mkdir(args.output_dir+os.sep+ticker.lower())

            to_date = date.fromisoformat(date_str)
            from_date = to_date - timedelta(days=NUM_DAYS)
            print(f"From Date: {from_date.isoformat()}")
            print(f"To Date: {to_date.isoformat()}")
            url = build_eodhd_url(ticker.lower(), from_date.isoformat(), to_date.isoformat(), "1000", "0")
            output_filename = args.output_dir+os.sep+ticker+os.sep+from_date.isoformat()+"_"+to_date.isoformat()+"_news.json"
            data = requests.get(url).json()
            with open(output_filename, "w+") as f:
                json.dump(data, f)
            



    return

if __name__ == '__main__':
    main()
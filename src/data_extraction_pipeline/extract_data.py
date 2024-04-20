import os, json
import argparse
from datetime import date, timedelta
import requests


# GLOBALS

# How many days before our chosen date do we fetch news articles from
NUM_DAYS = 2
TICKERS = {"AMZN", "NVDA", "F", "VZ"}

'''
Pipeline for extracting acquired data into separate documents 

'''

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
                        base_dir -> ticker_name -> to-date -> content_date.txt
    '''

    # Read json that contains tickers and dates being considered
    if not os.path.isdir(args.acquired_data_dir):
        print(f"Directory {args.acquired_data_dir} doesn't exist.")
        exit(1)

    tickers = [d for d in os.listdir(args.acquired_data_dir) if os.path.isdir(d)]
    ticker_dirs = [os.path.join(os.path.dirname(args.acquired_data_dir), t) for t in tickers]


    for t in ticker_dirs:
        

    return

if __name__ == '__main__':
    main()
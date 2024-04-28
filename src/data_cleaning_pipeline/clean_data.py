from techniques import CleaningTechniqueFactory
import os, sys, argparse

sys.path.append('..')

from utils import NewsArticleDataset


CLEANING_TECHNIQUES = ["lower_case", "remove_punctuation", "remove_special_chars", "remove_urls", \
                       "remove_stop_words", "stemming", "remove_named_entities"]



def load_dataset(dataset_path: str):
    # Check if pickle exists
    dataset = NewsArticleDataset()
    if os.path.isfile(dataset_path+ os.sep +'extracted_data.pkl'):
        dataset.load_from_cache(dataset_path+ os.sep +'extracted_data.pkl')

    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='ticker_data.json', help="Path to the JSON file that contains ticker names and dates.")
    parser.add_argument("--clean_technique", type=str, default='lower_case', help="Which cleaning technique to apply.")
    parser.add_argument("--output_dir", type=str, default='../../data/cleaned_data', help="Path to the directory where acquired data should be stored.")
    parser.add_argument("--debug", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")

    #Add any more arguments as and when needed
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_dir)
    factory = CleaningTechniqueFactory()
    clean_func = factory.generate_cleaning_technique(args.clean_technique)
    cleaned_data = map(clean_func, dataset['data'])
    cleaned_dataset = NewsArticleDataset(cleaned_data, dataset['labels'])

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.isdir(args.output_dir + os.sep + args.clean_technique):
        os.mkdir(args.output_dir + os.sep + args.clean_technique)

    cleaned_dataset.write_cache(args.output_dir + os.sep + args.clean_technique)
        


if __name__ == "__main__":
    main()
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

    return dataset.dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='../../data/extracted_data', help="Path to the pickled dataset.")
    parser.add_argument("-c", "--clean_technique", type=str, default='lower_case', help="Which cleaning technique to apply.")
    parser.add_argument("--output_dir", type=str, default='../../data/cleaned_data', help="Path to the directory where acquired data should be stored.")
    parser.add_argument("--debug", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("-s", "--sample", type=int, default=0, help="Writes out the first n samples of the original and cleaned data.")

    #Add any more arguments as and when needed
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_dir)
    factory = CleaningTechniqueFactory()
    clean_func = factory.generate_cleaning_technique(args.clean_technique)
    cleaned_data = list(map(clean_func, dataset['data']))
    cleaned_dataset = NewsArticleDataset(cleaned_data, dataset['labels'])

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.isdir(args.output_dir + os.sep + args.clean_technique):
        os.mkdir(args.output_dir + os.sep + args.clean_technique)

    cleaned_dataset.write_cache(args.output_dir + os.sep + args.clean_technique)

    if args.sample:
        for i in range(args.sample):
            with open(args.output_dir + os.sep + args.clean_technique + os.sep + 'sample_'+str(i)+'.txt', 'w') as f:
                f.write("Original Data:\n")
                f.write(dataset['data'][i])
                f.write('\n\nCleaned Data:\n')
                f.write(cleaned_data[i])


if __name__ == "__main__":
    main()
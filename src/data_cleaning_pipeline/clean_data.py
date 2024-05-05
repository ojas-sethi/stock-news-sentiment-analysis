from techniques import CleaningTechniqueFactory
import os, sys, argparse

sys.path.append('..')

from utils import NewsArticleDataset

def load_dataset(dataset_path: str, use_cache: bool):
    # Check if pickle exists
    dataset = NewsArticleDataset()
    if use_cache:
        if os.path.isfile(dataset_path+ os.sep +'extracted_data.pkl'):
            dataset.load_from_cache(dataset_path+ os.sep +'extracted_data.pkl')
        return dataset
    else:
        assert os.path.isdir(dataset_path+ os.sep + 'data') and \
               os.path.isdir(dataset_path+ os.sep + 'labels')
        dataset.load(dataset_path)
        return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='../../data/extracted_data', help="Path to the JSON file that contains ticker names and dates.")
    parser.add_argument("--clean_method", type=str, default='default', help="Which cleaning technique to apply. Default is to not apply any cleaning technique.")
    parser.add_argument("--output_dir", type=str, default='../../data/cleaned_data', help="Path to the directory where acquired data should be stored.")
    parser.add_argument("--debug", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("--write_cache", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("--load_from_cache", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")

    #Add any more arguments as and when needed
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_dir, args.load_from_cache)
    factory = CleaningTechniqueFactory()
    clean_func = factory.generate_cleaning_technique(args.clean_method)
    cleaned_data = list(map(clean_func, dataset.get_data()))
    print(len(cleaned_data))
    cleaned_dataset = NewsArticleDataset(cleaned_data, dataset.get_labels())

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.isdir(args.output_dir + os.sep + args.clean_method):
        os.mkdir(args.output_dir + os.sep + args.clean_method)

    if args.write_cache:
        cleaned_dataset.write_cache(args.output_dir + os.sep + args.clean_method)
    else:
        cleaned_dataset.write_dataset(args.output_dir + os.sep + args.clean_method)

if __name__ == "__main__":
    main()
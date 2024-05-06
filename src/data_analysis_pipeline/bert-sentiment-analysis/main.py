from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from urllib import request
from bs4 import BeautifulSoup
import csv, argparse, os, sys

sys.path.append("../..")

from utils import NewsArticleDataset


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

'''
Since we're using the pre-trained model, we need to map labels to the same indices as the original paper.
code from 
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='../../../data/cleaned_data/default', help="Path to the JSON file that contains ticker names and dates.")
    parser.add_argument("--clean_methods", type=str, default='default', help="Which cleaning technique to apply. Default is to not apply any cleaning technique.")
    parser.add_argument("--output_dir", type=str, default='../../data/cleaned_data', help="Path to the directory where acquired data should be stored.")
    parser.add_argument("--debug", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("--write_cache", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("--load_from_cache", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")

    #Add any more arguments as and when needed
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print("Directory doesn't exist. Please specify valid directory to read dataset from.")
        exit(1)

    dataset = NewsArticleDataset()
    dataset.load(args.dataset_dir)

    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
    with request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]
    print(labels)
    ## Don't compute gradients
    model.to(DEVICE).eval()
    scores = []

    for text in dataset.get_data():
    # TODO: remove this when done debugging
    #with open('article.txt', 'r') as f:
        #text = f.read()
        encoded_text = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

        result = model(encoded_text['input_ids'].to(DEVICE))

        score = torch.nn.functional.softmax(result.logits).detach().cpu().numpy()
        score_dict = {}

        for i in range(score.shape[1]):
            score_dict[labels[i]] = score[0, i]
        scores.append(score_dict)
    
    acc = dataset.compute_metrics(scores)

    print(acc)
    #print(f1)
    

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from urllib import request
from bs4 import BeautifulSoup
import csv, argparse, os, sys

sys.path.append("../..")

from utils import NewsArticleDataset
from transformers import TrainingArguments, Trainer
import numpy as np
import operator

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps') 



# Set global seed
np.random.seed(42)
torch.manual_seed(42)

torch.set_default_dtype(torch.float32)


def perform_test_train_split(dataset: NewsArticleDataset):
    shuffled_inds = np.random.choice(len(dataset.get_labels()), len(dataset.get_labels()), replace=False)
    train_end = int(0.8*len(shuffled_inds))
    train_inds = shuffled_inds[:train_end]
    test_inds = shuffled_inds[train_end:]

    return NewsArticleDataset([dataset.get_data()[ind] for ind in train_inds],\
                              [dataset.get_labels()[ind] for ind in train_inds]),\
           NewsArticleDataset([dataset.get_data()[ind] for ind in test_inds],\
                             [dataset.get_labels()[ind] for ind in test_inds])

'''
Since we're using the pre-trained model, we need to map labels to the same indices as the original paper.
code from 
'''

def train_model(model, train_set, labels, tokenizer):
    model.train()
    train_data = train_set.get_data()
    train_labels = torch.tensor([[t_label[l] for l in labels] for t_label in train_set.get_labels()]).to(DEVICE)

    optimizier = torch.optim.AdamW(params=model.parameters(), lr=2e-5, weight_decay=0.01)
    optimizier.zero_grad()

    for i in range(len(train_labels)):
        encoded_text = tokenizer(train_data[i], return_tensors="pt", truncation=True, max_length=512, padding=True)
        result = model(encoded_text['input_ids'].to(DEVICE))
        score = torch.nn.functional.softmax(result.logits, dim=-1)

        loss = torch.nn.functional.cross_entropy(score.squeeze(), train_labels[i])

        loss.backward()
        optimizier.step()
    return

def test_model(model, test_set, labels, tokenizer):
    model.eval()
    
    test_data = test_set.get_data()
    test_labels = torch.tensor([[t_label[l] for l in labels] for t_label in test_set.get_labels()])
    y_hat = []
    y = []

    for i in range(len(test_labels)):
        encoded_text = tokenizer(test_data[i], return_tensors="pt", truncation=True, max_length=512, padding=True)
        result = model(encoded_text['input_ids'].to(DEVICE))
        score = torch.nn.functional.softmax(result.logits).detach().cpu().numpy()
        y_hat.append(labels[int(np.argmax(score, axis=-1))])
        y.append(labels[int(np.argmax(test_labels[i], axis=-1))])

    acc = [int(y_hat[i]==y[i]) for i in range(len(y_hat))]

    print(f"Validation Accuracy: {sum(acc)/len(acc)}")

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='../../../data/cleaned_data/default', help="Path to the JSON file that contains ticker names and dates.")
    parser.add_argument("--clean_methods", type=str, default='default', help="Which cleaning technique to apply. Default is to not apply any cleaning technique.")
    parser.add_argument("--output_dir", type=str, default='./finetuned_models', help="Path to the directory where acquired data should be stored.")
    parser.add_argument("--debug", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("--load_from_cache", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")

    #Add any more arguments as and when needed
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print("Directory doesn't exist. Please specify valid directory to read dataset from.")
        exit(1)

    dataset = NewsArticleDataset()
    dataset.load(args.dataset_dir)
    train_set, test_set = perform_test_train_split(dataset)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
    with request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    model.to(DEVICE)

    if args.debug:
        print(f"Labels: {labels}")
        print(f"Using device: {DEVICE}")
    
    train_model(model, train_set, labels, tokenizer)

    test_model(model, test_set, labels, tokenizer)
    
    model.save_pretrained(f"{args.output_dir}", from_pt=True)



if __name__ == "__main__":
    main()
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from urllib import request
from bs4 import BeautifulSoup
import csv, argparse, os, sys
import pandas as pd
sys.path.append("../..")

from utils import NewsArticleDataset
from transformers import TrainingArguments, Trainer
import numpy as np
import json, urllib
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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
    train_end = int(10)
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

def test_model(model, test_set : NewsArticleDataset, labels :dict, tokenizer):
    model.to(torch.bfloat16).eval()
    
    test_data = test_set.get_data()
    test_labels = test_set.get_labels() #torch.tensor([[t_label[l] for l in labels] for t_label in test_set.get_labels()]).to(torch.bfloat16)
    y_hat = []
    y = []

    for i in range(len(test_labels)):
        encoded_text = tokenizer(test_data[i], return_tensors="pt", padding=True)
        result = model(encoded_text['input_ids'].to(DEVICE))
        score = torch.nn.functional.softmax(result.logits, dim=-1).detach()
        y_hat.append(labels[torch.argmax(score, axis=-1).item()])
        #print(f"Predicted Label: {y_hat[-1]}")
        #print(f"True Label: {test_labels[i]}")

    # Compute accuracy
    acc = accuracy_score(test_labels, y_hat)

    '''pos_count = 0
    neu_count = 0
    neg_count = 0
    for i in range(len(test_labels)):
        if np.argmax(test_labels[i], axis=-1).item() == 0:
            pos_count+=1
        elif np.argmax(test_labels[i], axis=-1).item() == 1:
            neu_count+=1
        elif np.argmax(test_labels[i], axis=-1).item() == 2:
            neg_count+=1
        else:
            print("WTF")
    print(f"pos_count: {pos_count}")
    print(f"neu_count: {neu_count}")
    print(f"neg_count: {neg_count}")'''
    
    # Compute Precision
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, y_hat, labels=list(labels.values()))
    

    # Compute Recall
    #recall = recall_score(y, y_hat, labels=labels)

    # Compute F1 Score
    #f1 = f1_score(y, y_hat, labels=labels)

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

    return acc, precision, recall, f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='../../../data/cleaned_data/default', help="Path to the JSON file that contains ticker names and dates.")
    parser.add_argument("--clean_method", type=str, default='default', help="Which cleaning technique to apply. Default is to not apply any cleaning technique.")
    parser.add_argument("--model_path", type=str, default='./finetuned_models', help="Path to the directory where acquired data should be stored.")
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
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", do_lower_case = False)

    labels=[]
    mapping_link = f"https://huggingface.co/ProsusAI/finbert/raw/main/config.json"
    with request.urlopen(mapping_link) as f:
        json_data = json.load(f)
        labels = {int(k): v for k, v in json_data["id2label"].items()}

    model.to(DEVICE)

    if args.debug:
        print(f"Labels: {labels}")
        print(f"Using device: {DEVICE}")
    
    #train_model(model, train_set, labels, tokenizer)

    acc, precision, recall, f1_score = test_model(model, dataset, labels, tokenizer)
    
    df = pd.DataFrame({'label':labels.values(), 
                       'accuracy': [acc]*len(precision), 
                       'precision': precision,
                       'recall': recall,
                       'f1_score': f1_score})
    df.to_csv(f"../sentiment_analysis_results/{args.dataset_dir.split('/')[-1]}.csv")

    #model.to(DEVICE).eval()
    #scores = []

    '''for text in dataset.get_data():
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
    
    acc = dataset.compute_metrics(scores)'''

    #print(acc)
    #print(f1)



if __name__ == "__main__":
    main()
    

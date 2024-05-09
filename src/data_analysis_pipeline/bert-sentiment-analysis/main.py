from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from urllib import request
from bs4 import BeautifulSoup
import csv, argparse, os, sys
from torch.utils.data import TensorDataset, DataLoader

sys.path.append("../..")

from utils import NewsArticleDataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import numpy as np
import operator

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:6')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps') 



# Set global seed
np.random.seed(42)
torch.manual_seed(42)

torch.set_default_dtype(torch.float32)


def perform_test_train_split(dataset: NewsArticleDataset):
    dataset = dataset.dataset
    data = dataset['data']
    labels = dataset['labels']

    numTrain = int(len(data) * 0.4)

    train_data = data[:numTrain]
    train_labels = labels[:numTrain]
    train = NewsArticleDataset(train_data, train_labels)

    test_data = data[numTrain:]
    test_labels = labels[numTrain:]
    test = NewsArticleDataset(test_data, test_labels)

    return train, test
    

'''
Since we're using the pre-trained model, we need to map labels to the same indices as the original paper.
code from 
'''

def train_model(model, train_set, labels, tokenizer):
    train_data = train_set.dataset['data']
    train_labels = train_set.dataset['labels']
    # train_labels = torch.tensor([[t_label[l] for l in labels] for t_label in train_set.get_labels()]).to(DEVICE)
    label_to_index = {"negative": 0, "neutral": 1, "positive": 2}

    # Convert string labels to numerical labels
    numerical_labels = [label_to_index[label] for label in train_labels]
    labels_tensor = torch.tensor(numerical_labels)

    encoded_data = tokenizer(train_data, return_tensors="pt", truncation=True, max_length=512, padding=True)
    dataset = TensorDataset(encoded_data['input_ids'], encoded_data['attention_mask'], labels_tensor)
    batch_size = 16
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

def test_model(model, test_set, labels, tokenizer):
    model.eval()

    test_data = test_set.dataset['data']
    test_labels = test_set.dataset['labels']
    # test_labels = torch.tensor([[t_label[l] for l in labels] for t_label in test_set.get_labels()]).to(DEVICE)
    label_to_index = {"negative": 0, "neutral": 1, "positive": 2}

    # Convert string labels to numerical labels
    numerical_labels = [label_to_index[label] for label in test_labels]
    labels_tensor = torch.tensor(numerical_labels)
    
    encoded_data = tokenizer(test_data, return_tensors="pt", truncation=True, max_length=512, padding=True)
    dataset = TensorDataset(encoded_data['input_ids'], encoded_data['attention_mask'], labels_tensor)
    batch_size = 16
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    acc = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    accuracy = sum(true_labels == predicted_labels) / len(true_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    print(f"Accuracy: {accuracy}")
    # Compute Precision, Recall, F1 Score
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='../../../data/extracted_data/extracted_data.pkl', help="Path to the JSON file that contains ticker names and dates.")
    parser.add_argument("--clean_methods", type=str, default='default', help="Which cleaning technique to apply. Default is to not apply any cleaning technique.")
    parser.add_argument("--output_dir", type=str, default='../../data/cleaned_data', help="Path to the directory where acquired data should be stored.")
    parser.add_argument("--debug", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("--write_cache", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("--load_from_cache", default=False, help="Setting flag to true disables api requests being sent out.", action="store_true")
    parser.add_argument("--mode", type=str, default='train', help="Mode to run the script in. Default is train.")
    #Add any more arguments as and when needed
    args = parser.parse_args()

    # if not os.path.isdir(args.dataset_dir):
    #     print("Directory doesn't exist. Please specify valid directory to read dataset from.")
    #     exit(1)

    dataset = NewsArticleDataset()
    dataset.load_from_cache(args.dataset_dir)
    

    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
    with request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    
    if args.mode == 'train':

        train_set, test_set = perform_test_train_split(dataset)
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        model.to(DEVICE)
    
        train_model(model, train_set, labels, tokenizer)

        test_model(model, test_set, labels, tokenizer)

        # save model
        model.save_pretrained("distilbertModel")
        tokenizer.save_pretrained("distilbertTokenizer")
    else:
        model = AutoModelForSequenceClassification.from_pretrained("distilbertModel", num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained("distilbertTokenizer")
        model.to(DEVICE)
        test_model(model, dataset, labels, tokenizer)
    
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
    

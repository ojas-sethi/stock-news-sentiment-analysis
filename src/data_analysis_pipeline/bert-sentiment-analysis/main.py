from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def read_text_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

filename = "article.txt"
text = read_text_from_file(filename)

tokens = tokenizer.encode(text, return_tensors="pt")

result = model(tokens)

sentiment_score = int(torch.argmax(result.logits)) + 1
sentiment_probability = torch.softmax(result.logits, dim=1)[0][sentiment_score - 1].item()

print("Sentiment Score:", sentiment_score)
# The sentiment score is a number between 1 and 5, where 1 is very negative and 5 is very positive

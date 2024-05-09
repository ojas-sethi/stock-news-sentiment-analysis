import sys
sys.path.append('..')
from utils import NewsArticleDataset
import pandas as pd

# load ../../data/forex.csv
df = pd.read_csv('../../data/forex/forex.csv')
# keep only the columns title, text and true_sentiment
df = df[['title', 'text', 'true_sentiment']]
# concatenate title and text into a new column called content
df['content'] = df['title'] + ' ' + df['text']
# drop the title and text columns
df.drop(columns=['title', 'text'], inplace=True)
#lower case the true_sentiment column
df['true_sentiment'] = df['true_sentiment'].str.lower()
print(df['true_sentiment'])
# extract content column as a list
data = df['content'].tolist()
labels = df['true_sentiment'].tolist()

# create a NewsArticleDataset object
dataset = NewsArticleDataset(data, labels)
dataset.write_cache('../../data/forex')
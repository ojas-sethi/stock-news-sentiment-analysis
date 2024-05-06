import os
import numpy as np
import pickle
import operator

from sklearn.metrics import f1_score

class NewsArticleDataset:
    def __init__(self, data_list = [], label_list = []) -> None:
        self.dataset = {'data': data_list, 'labels': label_list}

    def compute_metrics(self, results):
        assert len(results) == len(self.dataset['labels'])

        y_true_list = []
        y_hat_list = []
        acc = []
        for i, label in enumerate(self.dataset['labels']):
            y_true_list.append(max(self.dataset['labels'][i].items(), key=operator.itemgetter(1))[0])
            y_hat_list.append(max(results[i].items(), key=operator.itemgetter(1))[0])
            print(f"label: {max(self.dataset['labels'][i].items(), key=operator.itemgetter(1))[0]}")
            print(f"prediction: {max(results[i].items(), key=operator.itemgetter(1))[0]}")
            acc.append(int(max(results[i].items(), key=operator.itemgetter(1))[0] == \
                       max(self.dataset['labels'][i].items(), key=operator.itemgetter(1))[0]))
        y = np.array(y_true_list)
        y_hat = np.array(y_hat_list)
        accuracy = sum(acc)
        return accuracy

        

    def add_data(self, news_article):
        extract_labels = lambda  a : {v: a['sentiment'][k] \
                                      for k,v in {'neg':'negative', 'neu':'neutral', 'pos':'positive'}.items()}
        
        extract_news_data = lambda  a : "\n".join([a['title'], a['content']])

        self.dataset['data'].append(extract_news_data(news_article))
        self.dataset['labels'].append(extract_labels(news_article))

    def add_to_dataset(self, json_content):
        if isinstance(json_content, list):
            for news_article in json_content:
                self.add_data(news_article)
        else:
            self.add_data(news_article)

    def write_dataset(self, out_dir):
        # Create directory for dataset
        if not os.path.isdir(out_dir+os.sep+'data'):
            os.mkdir(out_dir+os.sep+'data')

        dataset_dir = out_dir+os.sep+'data'

        
        # Create directory for labels 
        if not os.path.isdir(out_dir+os.sep+'labels'):
            os.mkdir(out_dir+os.sep+'labels')

        label_dir = out_dir+os.sep+'labels'

        for i, doc in enumerate(self.dataset["data"]):
            # Write docs
            with open(dataset_dir + os.sep + str(i)+'.txt', 'w+') as f:
                f.write(doc)
            # Write labels
            with open(label_dir + os.sep + str(i)+'.pkl', 'wb') as f:
                pickle.dump(self.dataset["labels"][i], f, protocol=pickle.HIGHEST_PROTOCOL)

        return
                    
    def write_cache(self, out_dir):
        if not os.path.isdir(out_dir):
            print(f'Cannot write to {out_dir}. Directory doesn\'t exist.')
        
        tuple_list = []
        for i, doc in enumerate(self.dataset["data"]):
            tuple_list.append((doc, self.dataset['labels'][i]))

        with open(out_dir + os.sep + 'extracted_data.pkl', 'wb') as handle:
            pickle.dump(tuple_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_cache(self, filepath):
        with open(filepath, 'rb') as f:
            data_list = pickle.load(f)
            for data in data_list:
                self.dataset['data'].append(data[0])
                self.dataset['labels'].append(
                    {"polarity": data[1][0],
                     "neg": data[1][1],
                     "neu": data[1][2],
                     "pos": data[1][3]})
    
    def load(self, directory_path):
        labels_path = directory_path + os.sep + 'labels'
        data_path = directory_path + os.sep + 'data'

        num_data_samples = len([name for name in os.listdir(data_path) \
                                if os.path.isfile(os.path.join(data_path, name))])
        num_labels = len([name for name in os.listdir(labels_path) \
                          if os.path.isfile(os.path.join(labels_path, name))])

        assert num_data_samples == num_labels

        for i in range(num_data_samples):
            with open(data_path + os.sep + str(i)+'.txt', 'r') as f:
                self.dataset['data'].append(f.read())
            with open(labels_path + os.sep + str(i)+'.pkl', 'rb') as f:
                self.dataset['labels'].append(pickle.load(f))
        return
                
    def get_data(self):
        return self.dataset['data']
    
    def get_labels(self):
        return self.dataset['labels']


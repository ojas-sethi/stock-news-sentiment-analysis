import os
import numpy as np

class NewsArticleDataset:
    def __init__(self) -> None:
        self.dataset = {'data': [], 'labels': []}

    def add_data(self, news_article):
        extract_labels = lambda  a : np.array([a['sentiment']["polarity"], \
                                               a['sentiment']["neg"], \
                                               a['sentiment']["neu"], \
                                               a['sentiment']["pos"]])
        
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
        with open(label_dir + os.sep + 'labels.npy', 'wb') as f:
            np.save(f, np.array(self.dataset['labels']))

        return
                    
    def write_cache(self, out_dir):
        if not os.path.isdir(out_dir):
            print(f'Cannot write to {out_dir}. Directory doesn\'t exist.')
        
        tuple_list = []
        for i, doc in enumerate(self.dataset["data"]):
            tuple_list.append((doc, self.dataset['labels'][i]))

        import pickle
        with open(out_dir + os.sep + 'extracted_data.pkl', 'wb') as handle:
            pickle.dump(tuple_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

class NewsArticleDataset:
    def __init__(self) -> None:
        self.dataset = {'data': [], 'labels': []}
        

    def extract_news_data(self, news_article): 
        return "\n".join([news_article['title'], news_article['content']])

    def add_data(self, news_article):
        self.dataset['data'].append(self.extract_news_data(news_article))
        self.dataset['labels'].append(news_article['sentiment'])

    def add_to_dataset(self, json_content):
        if isinstance(json_content, list):
            for news_article in json_content:
                self.add_data(news_article)
        else:
            self.add_data(news_article)
                    

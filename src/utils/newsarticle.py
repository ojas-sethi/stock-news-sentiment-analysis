
class NewsArticle:
    def __init__(self, title: str, date: str, content: str,\
                 open_price: float = None, close_price: float=None) -> None:
        self.title = title
        self.date = date
        self.content = content
        self.open_price = open_price
        self.close_price = close_price
        
    def set_open_price(self, open_price):
        self.open_price = open_price
    
    def set_close_price(self, close_price):
        self.close_price = close_price

    def build_dataset():
        return
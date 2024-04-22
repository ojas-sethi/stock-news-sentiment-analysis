
class NewsArticle:
    def __init__(self, title: str, date: str, content: str) -> None:
        self.title = title
        self.date = date
        self.content = content
        self.open_price = None
        self.close_price = None

    def __init__(self, title: str, date: str, content: str, open_price: float, close_price: float) -> None:
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
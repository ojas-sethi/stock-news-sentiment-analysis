import urllib.request
import bs4, urllib


CLASS_NAME = "clearfix text-formatted field field--name-body " +\
             "field--type-text-with-summary field--label-hidden field__item"
URL = "https://www.finra.org/registration-exams-ce/qualification-exams/terms-and-acronyms"

if __name__ == "__main__":
    html_content = urllib.request.urlopen(URL)

    soup = bs4.BeautifulSoup(html_content, "lxml")
    # We can flatten the list out. Maintaining hieriarchy is useless
    term_list = []
    for class_elements in soup.find_all("div", {"class": CLASS_NAME}):
        if class_elements.find_all("li"):
            term_list = [el.get_text() for el in class_elements.find_all("li")]

    print(term_list)
    
    


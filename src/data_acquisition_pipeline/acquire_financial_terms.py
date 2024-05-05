from urllib import request
import bs4, urllib, sys, json


CLASS_NAME = "clearfix text-formatted field field--name-body " +\
             "field--type-text-with-summary field--label-hidden field__item"
URL = "https://www.finra.org/registration-exams-ce/qualification-exams/terms-and-acronyms"

if __name__ == "__main__":
    assert len(sys.argv) <=2
    output_path = "../data_cleaning_pipeline/financial_terms.json"
    if len(sys.argv) == 2:
        output_path = sys.argv[1]

    html_content = request.urlopen(URL)

    soup = bs4.BeautifulSoup(html_content, "lxml")
    # We can flatten the list out. Maintaining hieriarchy is useless
    term_list = []
    for class_elements in soup.find_all("div", {"class": CLASS_NAME}):
        if class_elements.find_all("li"):
            term_list = [el.get_text() for el in class_elements.find_all("li")]

    financial_abbr_map = dict()
    for term in term_list:
        term_arr = term.split('(')
        if len(term_arr) == 1:
            continue
        abbr = term.split('(')[1].split(')')[0]
        full_form = term.split('(')[0].strip()
        financial_abbr_map[abbr] = full_form
    
    with open(output_path, 'w') as f:
        json.dump(financial_abbr_map, f, indent=4)


    

    
    


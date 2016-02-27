'''
    Beautifier is a module that does NLP on HTML files from the data for this project
'''

from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize as wt
from nltk.stem import PorterStemmer
from textblob import TextBlob

stop_words = []
stemmer = PorterStemmer()

with open('res/StopWords.txt', 'rb') as infile:
    stop_words_file = infile.read().decode('UTF-8')
    stop_words = set(wt(stop_words_file))

def get_soup_text(soup):
    for scriptdata in soup.findAll(["script", "style"]):
        scriptdata.extract()
    return soup.text.lower()

def remove_stop_words_and_stem(soup):
    result = get_soup_text(soup)
    tokenized_text = wt(result)
    filtered_array = []
    for word in tokenized_text:
        # Eliminating the stopwords
        if word not in stop_words and word.isalpha():
        # Only considering alphabetical strings
            filtered_array.append(stemmer.stem(word))
    return TextBlob(' '.join(filtered_array))


def main():
    html_doc = """
        <html><head><title>The Dormouse's story</title></head>
        <body>
        <p class="title"><b>The Dormouse's story</b></p>

        <p class="story">Once upon a time there were three little sisters; and their names were
        <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
        <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
        <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
        and they lived at the bottom of a well.</p>

        <p class="story">...</p>
        """

    soup = bs(html_doc, 'html.parser')
    result = remove_stop_words_and_stem(soup)
    print(result)


if __name__ == '__main__':
    main()
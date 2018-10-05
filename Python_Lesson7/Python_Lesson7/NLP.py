import requests
import urllib
import os
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')

#URL link for the url_code
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

# #Getting the requets from web page
html_page = requests.get(url)

#
# # using beautiful soup for html parser
soup = BeautifulSoup(html_page.text, "html.parser")

[s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
visible_text = soup.getText()#gets the above mentioned from the given web page

import io
with io.open(r'/Users/pandala/Desktop/Python_Lesson7/icp7.txt ', "w+", encoding="utf-8") as f:
    f.write(visible_text)#after getting the data from the web page saves it an text file where we perform following operations
    f.close()


import  nltk
from nltk.tokenize import  word_tokenize, sent_tokenize,wordpunct_tokenize#this one is for tokenization

from nltk.stem import PorterStemmer, LancasterStemmer#this one is for stemming
from nltk import pos_tag, ne_chunk# this one is for pos
from nltk.util import  ngrams#this one is for n grams
from collections import  Counter
from nltk.stem import WordNetLemmatizer#this one is for lemmatization



j = []
# #
with io.open(r'/Users/pandala/Desktop/Python_Lesson7/icp7.txt', "r", encoding="utf-8") as fo:
    print("sentence tokenization")
    for i in fo.readlines():#the below print statements accomplishes the given task for us
         tokenization = word_tokenize(i)
         print(tokenization)
         print(pos_tag(i))
         print(Counter(ngrams(word_tokenize(i), 3)))
         print(ne_chunk(pos_tag(wordpunct_tokenize(i))))

         stemmer = LancasterStemmer()
        # print(stemmer.stem(i.split()))
         lemmetizer = WordNetLemmatizer()
         for data in tokenization:
             print("Lemmetizer for word: ", data,": " ,lemmetizer.lemmatize(data))
             print("Stemmer for word: ", data, ": ", stemmer.stem(data))
import requests
import urllib.request
import os
from bs4 import BeautifulSoup

link_list=[]


html_page = requests.get("https://en.wikipedia.org/wiki/Deep_learning")



soup = BeautifulSoup(html_page.text, "html.parser")


print(soup.title.string)

for i in soup.find_all('a'):
    link_list.append()
    print(i.get('href'))

print(link_list)


th_list = [ ]

titles = []

print(titles)
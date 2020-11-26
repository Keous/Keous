import urllib3
import certifi
from bs4 import BeautifulSoup
import threading
from queue import Queue
import re
from collections import OrderedDict,Counter
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import time
from itertools import chain
from .article import Article,Collection
import feedparser


root_urls=[
'https://www.breitbart.com/politics/',
'https://www.foxnews.com/politics/',
'https://www.washingtontimes.com/news/politics/',
'https://thefederalist.com/category/politics/',
'https://www.nationalreview.com/politics-policy/',
'https://nypost.com/tag/politics/',
'https://www.theblaze.com/politics/',
#'https://www.theamericanconservative.com/web-categories/politics/',
#'https://www.marketwatch.com/economy-politics/', #needs work


'https://www.npr.org/sections/politics/',
'https://thehill.com/',

'https://www.cbsnews.com/politics/',
'https://www.vox.com/politics/',
#'https://www.newsweek.com/politics/', 
'https://www.cnn.com/politics/', 
'https://www.nytimes.com/section/politics/',
'https://www.theatlantic.com/politics/',
'https://www.nbcnews.com/politics/',
'https://www.huffpost.com/news/topic/us-politics/',
'https://www.motherjones.com/politics/',
'https://www.washingtonpost.com/politics/',
]

def get_urls(root_url,http,browser=False,max_n=15):
    urls = []
    #any([base in root_url for base in {'cnn.com','theamericanconservative.com'}])
    if browser == True: #needs to use a browser to scrape content
        opts = Options()
        opts.headless = True
        driver=webdriver.Firefox(options=opts)
        driver.get(root_url)
        html = driver.page_source
        soup = BeautifulSoup(html,'html.parser')
        time.sleep(1)
        driver.close()
    else:
        try:
            response = http.request('GET',root_url)
        except urllib3.exceptions.MaxRetryError:
            print('Error scraping '+root_url)
            return []
        soup = BeautifulSoup(response.data,'html.parser')
    if any([base in root_url for base in {'vox.com','nationalreview.com','thefederalist.com','nypost.com','nytimes.com','npr.org'}]):
        pattern = '/2020/' #special pattern, NEEDS TO BE CHANGED YEARLY
    elif any([base in root_url for base in {'theblaze.com','cbsnews.com'}]):
        pattern = '/news/'
    elif any([base in root_url for base in {'theamericanconservative.com'}]):
        pattern = '/articles/'
    elif any([base in root_url for base in {'newsweek.com'}]):
        pattern = ''
    elif 'huffpost.com' in root_url: #scrape RSS feed because hard to use rule-based without CSS
        feed_url = 'https://www.huffpost.com/section/politics/feed'
        urls = [article.link for article in feedparser.parse(feed_url)['entries']][:max_n]
        return urls
    elif 'thehill.com' in root_url: #scrape RSS because too much content to sift
        feed_url = 'https://thehill.com/rss/syndicator/19109'
        urls = [article.link for article in feedparser.parse(feed_url)['entries']][:max_n]
        return urls
    elif 'cnn.com' in root_url:
        feed_url = 'http://rss.cnn.com/rss/cnn_allpolitics.rss'
        urls = [article.link for article in feedparser.parse(feed_url)['entries']][:max_n]
        return urls
    else:
        sub_root = re.search('http(s?)://(.+?)/(.+?)/',root_url).group(3) #get sub root, i.e in foxnews/politics/ gets /politics/'
        pattern = '/{}/'.format(sub_root)


    naked_root_url = re.search('http(s?)://(.+?)/(.+?)/',root_url).group(2) #get root without extra stuff, i.e https://foxnews.com/politics becomes foxnews.com    
    for link in soup.find_all('a'):
        url = link.get('href')
        if url is None:
            continue    
        m = re.search(pattern,url)
        if m and url!=root_url:
            if url[:4]=='http': #if abesolute link
                if naked_root_url in url: #if on the same root url
                    urls.append(url)
            if url[0]=='/': #if relative link, make abesolute link
                urls.append('https://'+naked_root_url+url)
    return list(OrderedDict.fromkeys(urls))[:max_n] #cute trick to remove duplicates while preserving order


def scrape_all(root_urls=root_urls[:1],max_n=15):
    urls = []
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where(),timeout=12.0)
    for root_url in root_urls:
            urls.append(get_urls(root_url,http=http,max_n=max_n))
    flat=list(chain.from_iterable(urls))
    print(len(flat))
    c = Collection(flat,max_threads=100,http=http,verbose=0)
    print(len(c))
    return c
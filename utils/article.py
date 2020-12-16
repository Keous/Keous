import urllib3
import certifi
from bs4 import BeautifulSoup
import pickle
import threading
from queue import Queue
import feedparser
from unicodedata import normalize
import warnings
from itertools import chain
import traceback
import json
import msgpack
import numpy as np

class Article(object):
	'''Container for articles, source based parsing for text and paragraphs'''
	def __init__(self,data=None,http=None):
		if data is None:
			return None
		if isinstance(data,feedparser.FeedParserDict): #if all feed objects
			self.url = data['link']
			self.title = data['title']

		elif isinstance(data,str): #if url passed
			self.url = data

		elif isinstance(data,dict): #json-like data
			self.load_from_dict(data)

		else:
			raise TypeError('Either enter a feed object, a url, or json-like attributes')


		if http is not None:
			self.http = http
			
		if not hasattr(self,'source'):
			self.get_source()

		if not hasattr(self,'paras'):
			self.parse()


	def load_from_dict(self,data):
		if 'ent_sents' in data:
			ent_sents = data.pop('ent_sents')
			self.ent_sents = np.array(ent_sents)
		self.__dict__.update(data)


	def save_to_dict(self):
		data = {
				'paras':self.paras,
				'title':self.title,
				'source':self.source,
				'url':self.url,
				}
		if hasattr(self,'ents'):
			data['ents']=self.ents

		if hasattr(self,'ent_sents'):
			data['ent_sents']=self.ent_sents.tolist() #msgpack does not support arrays
		
		return data


	def save(self,file):
		data = self.save_to_dict()
		with open(file,'wb') as f:
			f.write(msgpack.packb(data))

	def load(self,file):
		with open(file,'rb') as f:
			data = msgpack.unpackb(f.read())
		self.load_from_dict(data)
		return self


	def get_source(self):
		if 'https://www.foxnews.com' in self.url:
			self.source = 'Fox News'
		elif 'https://www.nytimes.com' in self.url:
			self.source = 'New York Times'
		elif 'https://www.nbcnews.com' in self.url:
			self.source = 'NBC News'
		elif 'https://www.huffpost.com' in self.url:
			self.source = 'Huffington Post'
		elif 'https://www.cbsnews.com' in self.url:
			self.source = 'CBS News'
		elif 'https://www.npr.org' in self.url:
			self.source = 'NPR'
		elif 'https://www.cnn.com' in self.url or 'http://rss.cnn.com' in self.url:
			self.source = 'CNN'
		elif 'https://thehill.com' in self.url:
			self.source = 'Hill'
		elif 'washingtontimes.com' in self.url:
			self.source = 'Washington Times'
		elif 'https://www.breitbart.com' in self.url:
			self.source = 'Breitbart'
		elif 'https://thefederalist.com' in self.url:
			self.source = 'Federalist'
		elif 'https://www.nationalreview.com' in self.url:
			self.source = 'National Review'
		elif 'https://www.theblaze.com' in self.url:
			self.source = 'Blaze'
		elif 'https://nypost.com' in self.url:
			self.source = 'New York Post'
		elif 'https://www.theamericanconservative.com' in self.url:
			self.source = 'American Conservative'
		elif 'https://www.marketwatch.com' in self.url:
			self.source = 'Market Watch'
		elif 'https://www.motherjones.com' in self.url:
			self.source = 'Mother Jones'
		elif 'https://www.theatlantic.com' in self.url:
			self.source = 'Atlantic'
		elif 'https://www.vox.com' in self.url:
			self.source = 'Vox'
		elif 'https://www.newsweek.com' in self.url:
			self.source = 'Newsweek'
		elif 'https://www.washingtonpost.com' in self.url:
			self.source = 'Washington Post'
		else:
			self.source = 'Unrecognized'
			warnings.warn("Unrecognized source! Url: {}".format(self.url))
		   
	def parse(self):
		if not hasattr(self,'http'):
			self.http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where(),headers={'User-Agent': 'Mozilla/5.0'}) #retries=retries,timeout=timeout
			
		if self.source == 'Fox News':
			self.Fox_News()
		elif self.source == 'New York Times':
			self.New_York_Times()
		elif self.source == 'NBC News':
			self.NBC_News()
		elif self.source == 'Huffington Post':
			self.Huffington_Post()
		elif self.source == 'CBS News':
			self.CBS_News()
		elif self.source == 'NPR':
			self.NPR()
		elif self.source == 'CNN':
			self.CNN()
		elif self.source == 'Hill':
			self.Hill()
		elif self.source == 'Washington Times':
			self.Washington_Times()
		elif self.source == 'Breitbart':
			self.Breitbart()
		elif self.source == 'Federalist':
			self.Federalist()
		elif self.source == 'National Review':
			self.National_Review()
		elif self.source == 'Blaze':
			self.Blaze()
		elif self.source == 'New York Post':
			self.NY_Post()
		elif self.source == 'American Conservative':
			self.American_Consv()
		elif self.source == 'Market Watch':
			self.Market_Watch()
		elif self.source == 'Mother Jones':
			self.Mother_Jones()
		elif self.source == 'Atlantic':
			self.Atlantic()
		elif self.source == 'Vox':
			self.Vox()
		elif self.source == 'Newsweek':
			self.Newsweek()
		elif self.source == 'Washington Post':
			self.Washington_Post()
		elif self.source == 'Unrecognized':
			self.paras = None #disgregard when making into collection
		else:
			raise ValueError("Unrecognized source! We support the following sources: {}. Url: {}".format(recognized_sources,self.url))
		if self.paras is not None:
			self.length = len(self.paras)

	def Fox_News(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)

		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div', attrs={'class':'article-body'}) 
		paras = body.find_all('p')
		
		self.paras = [normalize('NFKD',para.get_text()) for para in paras]
		self.title = soup.find('h1',attrs={'class':'headline'}).text
		self.image_url = list(list(soup.find('div',attrs={'class':'m video-player'}).children)[0].children)[0].attrs['src']

	def New_York_Times(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
	   
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find_all('p',class_='css-158dogj evys1bk0')

		
		self.paras = [normalize('NFKD',para.get_text()) for para in body]
		self.title = soup.find('h1',attrs={'itemprop':'headline'}).text

	def NBC_News(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
	   
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find_all('p',class_='endmarkEnabled')
		self.paras = [normalize('NFKD',para.get_text()) for para in body]
		self.title = soup.find('h1',attrs={'class':'article-hero__headline f8 f9-m fw3 mb3 mt0 f10-xl founders-cond lh-none'}).text
		self.image_url = list(soup.find('picture',attrs={'class':'article-hero__main-image'}).children)[-1].attrs['src']

	def Huffington_Post(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
	   
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find_all('div',attrs={'class':'yr-content-list-text'})
		
		self.paras = [normalize('NFKD',para.get_text()) for para in body]
		self.title = soup.find('h1',attrs={'class':'headline__title'}).text

	def CBS_News(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
	   
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('section', attrs={'class':'content__body'})
		
		p_tags = body.find_all('p')
		self.paras = [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1',attrs={'class':'content__title'}).text
		self.image_url = list(soup.find('span',attrs={'class':'img embed__content'}).children)[0].attrs['src']
			
	def NPR(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)

		soup = BeautifulSoup(response.data,'html.parser')
		soup.find('div',attrs={'class':'bucketwrap image large'}).decompose() #remove image caption
		body = soup.find('div', attrs={'id':'storytext'})
		p_tags = body.find_all('p')
		
		self.paras = [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1').text

	def CNN(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
	   
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find_all(class_='zn-body__paragraph')

		self.paras= [normalize('NFKD',para.get_text()) for para in body]
		self.title = soup.find('h1',attrs={'class':'pg-headline'}).text

	def Hill(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		for span in soup.find_all('span',attrs={'class':'rollover-people-block'}):
			span.decompose() #remove rollover blocks with unrelated text
		body = soup.find('div', attrs={'class':['field-item','even'],'property':'content:encoded'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1',attrs={'class':'title'}).text

	def Washington_Times(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div', attrs={'class':'bigtext'})
		
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1',attrs={'class':'page-headline'}).text
		
	def Breitbart(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div', attrs={'class':'entry-content'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1').text
		image_url = soup.find('img',attrs={'class':'wp-post-image'})
		if image_url is not None:
			self.image_url = image_url.attrs['src']
		elif image_url is None:
			self.image_url = None

	def Federalist(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div', attrs={'class':'entry-content'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h2',attrs={'class':'entry-title'}).text

	def National_Review(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div', attrs={'class':'article-content'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1').text

	def Blaze(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div', attrs={'class':'body-description'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1').text

	def NY_Post(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div', attrs={'class':'entry-content entry-content-read-more'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1').text

	def American_Consv(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div', attrs={'class':'c-single-blog__content c-content'})
		p_tags = body.find_all('p')
		self.paras = [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1').text

	def Market_Watch(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div', attrs={'class':'article__body article-wrap at16-col16 barrons-article-wrap'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1',attrs={'class':'article__headline'}).text

	def Mother_Jones(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('article', attrs={'class':'entry-content'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1',attrs={'class':'entry-title'}).text

	def Atlantic(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('article', attrs={'id':'main-article'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1',attrs={'class':'c-article-header__hed'}).text

	def Vox(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div',attrs={'class':'c-entry-content'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1',attrs={'class':'c-page-title'}).text
		
	def Newsweek(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div',attrs={'class':'article-body v_text paywall'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1',attrs={'class':'title'}).text

	def Washington_Post(self):
		#http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
		response = self.http.request('GET',self.url)
		soup = BeautifulSoup(response.data,'html.parser')
		body = soup.find('div',attrs={'class':'article-body'})
		p_tags = body.find_all('p')
		self.paras= [normalize('NFKD',para.get_text()) for para in p_tags]
		self.title = soup.find('h1').text

	def __len__(self):
		return len(self.paras)

	def __iter__(self):
		return iter(self.paras)
   
	def __getitem__ (self, i):
		return self.paras[i]
   
	def __setitem__ (self,index,value):
		self.paras[index] = value

	def __contains__ (self,item):
		if isinstancence(item,list):
			if all([True for sub_item in item if sub_item in self.paras]) == True:
				return True
			else:
				return False
		elif isinstance(item,str):
			for para in self.paras:
				if item in para:
					return True
			return False
		else:
			raise TypeError('Unsupported containment check for type {}'.format(type(item)))

	def __add__ (self,other):
		if isinstance(other,Article):
			return Collection((self,other))
		elif isinstance(other,Collection):
			print('Did you mean {0}+{1} instead of {1}+{0}? Performed operation {0}+{1}'.format(type(m),type(a)))
			return other+self
		else:
			raise TypeError('Unsupported operand betwee types Article and {}'.format(type(other)))
		 

	def __repr__ (self):
		return self.title

	def __str__(self):
		return self.text()


	def text(self,sep='\n'):
			return sep.join(self.paras)+sep

	def get_ents(self,nlp):
		'''Return named ents'''
		accepted = ['PERSON','NORP','ORG','GPE','PRODUCT','EVENT','LAW']

		doc=nlp(self.text(sep='\n'))

		self.ents = [[ent.text for ent in doc.ents if ent.label_ in accepted] for doc in nlp.pipe(self.paras)]
		return self

	def calculate_reward(self,minimum_wage=12,split=False,time_per_q=7.5,wpm=250):
		'''Calculate reward for an article on Mturk'''
		cost=0
		assert hasattr(self,'ents')
		chars = len(self.text(''))
		questions = len(list(chain.from_iterable(self.ents)))+1 #add 1 for question at the end
		cost_chars=chars*minimum_wage/(wpm*60*6) #250 words per minute, 6 chars per word
		cost_questions=questions*minimum_wage*time_per_q/60/60 #takes 7.5 seocnds to answer a question
		if split==False:
			return cost_chars+cost_questions
		elif split==True:
			return (cost_chars,cost_questions)

	def to_json(self,chars_in_description=350):
		return {
		'url':self.url,
		'title':self.title,
		'description': self.text()[:chars_in_description], #self.paras[0][:chars_in_description]
		'source':self.source,
		}





class Collection(object):
	'''Used for a collection of articles. Can either pass urls to be threaded, or can pass articles'''
   
	def __init__(self,data=None,max_threads=30,http=None,verbose=1):
		if http is None:
			self.http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where(),headers={'User-Agent': 'Mozilla/5.0'})
		else:
			self.http = http

		self.verbose = verbose
		
		if data == None: #if called empty
			self.articles = []

		elif all([isinstance(a,dict) for a in data]): #json-like data
			self.load_from_dict(data)

		elif all([isinstance(a,Article) for a in data]): #if all articles
			self.articles = list(data)

		elif all([isinstance(a,feedparser.FeedParserDict) for a in data]):# if all feed objects
			self.articles = self.thread(data,max_threads)

		elif all([isinstance(a,str) for a in data]): #if all urls
			self.articles = self.thread(data,max_threads)
			
		else:
			raise TypeError('Either use all feeds, all urls, or all articles')
	 
	def thread(self,urls,max_threads):
		if len(urls) >= max_threads:
			n_threads = max_threads
		elif len(urls) < max_threads:
			n_threads = len(urls)
		threads = []
		q=Queue()
		threaded_articles = []
		for i in range(n_threads):
			t = threading.Thread(target=self.worker,args=(q,threaded_articles))
			t.start()
			threads.append(t)
		for url in urls:
			q.put(url)
		q.join()

		for i in range(n_threads):
			q.put(None)
		for t in threads:
			t.join()
		return [t for t in threaded_articles if t.paras != None and len(t.paras)!=0]

	def worker(self,q,threaded_articles):
		while True:
			url = q.get()
			if url is None:
				break
			try:
				threaded_articles.append(Article(url,http=self.http))
			except Exception as e: #need to add more conditions
				if self.verbose==2:
					print('Error scraping {}, got error {} with traceback {}'.format(url,e,traceback.format_exc()))
				elif self.verbose==1:
					print('Error scraping {}'.format(url))
				elif self.verbose==0:
					pass
			q.task_done()


   
	def add(self,other):
		if isinstance(other,Article):
			self.articles.add(other)
		else:
			raise TypeError('Adding type {} unsupported. Must add type Article'.format(type(other)))


	def __contains__ (self,item):
		if isinstance(item,Article):
			return item in self.articles
		elif isinstance(item,Collection):
			return list(set(item.articles).issubset(set(self.articles)))
		else:
			raise TypeError('Unsupported containment check for type {}'.format(type(item)))
   
	def __add__ (self,other):
		if isinstance(other,Article):
			return Collection(self.articles+other)
		elif isinstance(other,Collection):
			return Collection(self.articles+other.articles)
		else:
			raise TypeError('Unsupported operand betwee types Collection and {}'.format(type(other)))
		 
	def __sub__ (self,other):
		if isinstance(other,Article):
			if other in self:
				articles =  self.articles.copy()
				articles.remove(other)
				return Collection(articles)
			elif other not in self:
				raise ValueError('{} not in {}'.format(other,self))
		elif isinstance(other,Collection):
			if other in self:
				articles = set(self.articles).difference(set(other.articles))
				return Collection(articles)
			elif other not in self:
				raise ValueError('{} not in {}'.format(other,self))
		else:
			raise TypeError('Unsupported operand between types Collection and {}'.format(type(other)))

	def __len__ (self):
		return len(self.articles)

	def __iter__(self):
		return iter(self.articles)

	def __getitem__ (self,i):
		articles = self.articles[i]
		if isinstance(articles,list):
			return Collection(articles)
		elif isinstance(articles,Article):
			return articles

	def remove(self,other):
		if isinstance(other,Article):
			self.articles.remove(other)
		else:
			raise TypeError('Unsupported operand between types Collection and {}'.format(type(other)))
					

	def get_ents(self,nlp):
		for article in self.articles:
			article.get_ents(nlp=nlp)
		return self

	def load_predicted_sentiments(self,pred):
		assert sum([len(list(chain.from_iterable(a.ents))) for a in self]) == len(pred)
		tot=0
		for a in self:
			n=len(list(chain.from_iterable(a.ents)))
			a.ent_sents = pred[tot:tot+n]
			tot+=n


	def load_from_dict(self,data):
		self.articles = [Article(a_data) for a_data in data]
		return self

	def save_to_dict(self):
		data = [a.save_to_dict() for a in self.articles]
		return data

	def load(self,file):
		with open(file,'rb') as f:
			data = msgpack.unpackb(f.read())
		self.load_from_dict(data)
		return self

	def save(self,file):
		data = self.save_to_dict()
		with open(file,'wb') as f:
			f.write(msgpack.packb(data))




def get_source(url):
	if 'https://www.foxnews.com' in url:
		return 'Fox News'
	elif 'https://www.nytimes.com' in url:
		return 'New York Times'
	elif 'https://www.nbcnews.com' in url:
		return 'NBC News'
	elif 'https://www.huffpost.com' in url:
		return 'Huffington Post'
	elif 'https://www.cbsnews.com' in url:
		return 'CBS News'
	elif 'https://www.npr.org' in url:
		return 'NPR'
	elif 'https://www.cnn.com' in url:
		return 'CNN'
	elif 'https://thehill.com' in url:
		return 'Hill'
	elif 'washingtontimes.com' in url:
		return 'Washington Times'
	elif 'https://www.breitbart.com' in url:
		return 'Breitbart'
	elif 'https://thefederalist.com' in url:
		return 'Federalist'
	elif 'https://www.nationalreview.com' in url:
		return 'National Review'
	elif 'https://www.theblaze.com' in url:
		return 'Blaze'
	elif 'https://nypost.com' in url:
		return 'New York Post'
	elif 'https://www.theamericanconservative.com' in url:
		return 'American Conservative'
	elif 'https://www.marketwatch.com' in url:
		return 'Market Watch'
	elif 'https://www.motherjones.com' in url:
		return 'Mother Jones'
	elif 'https://www.theatlantic.com' in url:
		return 'Atlantic'
	elif 'https://www.vox.com' in url:
		return 'Vox'
	elif 'https://www.newsweek.com' in url:
		return 'Newsweek'
	elif 'https://www.washingtonpost.com' in url:
		return 'Washington Post'


class Pairs:
	'''Holds article pairs'''
	def __init__(self,pairs):
		new_pairs = []
		for p in pairs:
			article_a = p[3]
			article_b = p[4]
			if article_a.image_url is not None:
				image_url = article_a.image_url 
			elif article_b.image_url is not None:
				image_url = article_a.image_url
			elif article_a.image_url is None and article_b.image_url is None:
				image_url = 'https://www.keous.ai/static/images/noimage.png'
			new_pairs.append((image_url,)+p) #insert new item at the start
		self.pairs = new_pairs

	
	def to_json(self,chars_in_description=350):
		jsonized=[]
		for pair in self.pairs:
			image_url,dis_sim,cos_sim,euc_dist,a1,a2 = pair
			pair_json=[image_url,float(dis_sim),float(cos_sim),float(euc_dist),a1.to_json(chars_in_description=chars_in_description),a2.to_json(chars_in_description=chars_in_description)]
			jsonized.append(pair_json)
		return json.dumps(jsonized)
if __name__=='__main__':
	import sys
	import pytz
	import datetime
	import subprocess
	import os

	base = datetime.datetime.now(pytz.utc).strftime('%m-%d-%y-')
	dir_path=os.path.dirname(os.path.realpath(__file__))

	def scrape_urls():
		from .utils.scrape import add_urls
		subprocess.run('gsutil cp gs://keous-model-files/data/urls.txt .',shell=True,check=False)
		add_urls('urls.txt')
		subprocess.run('gsutil cp urls.txt gs://keous-model-files/data/urls.txt',shell=True,check=False)

	def build_collection():
		from .utils.scrape import scrape_file
		#import spacy
		#import neuralcoref

		#nlp=spacy.load('en_core_web_lg')
		#neuralcoref.add_to_pipe(nlp)

		subprocess.run('gsutil cp gs://keous-model-files/data/urls.txt .',shell=True,check=False)
		scrape_file('urls.txt',base+'collection.msgpack',get_ents=False,clear=False)
		subprocess.run('gsutil cp {} gs://keous-model-files/data/{}/'.format(base+'collection.msgpack',base),shell=True,check=False)
		subprocess.run('gsutil cp urls.txt gs://keous-model-files/data/urls.txt',shell=True,check=False)


	def cluster():
		from .models.bert_model import MyModel,read_out
		from .utils.article import Collection
		import torch

		subprocess.run('gsutil cp gs://keous-model-files/data/{}/{} .'.format(base,base+'collection.msgpack'),shell=True,check=False)
		c=Collection().load(base+'collection.msgpack')
		model_file = os.path.join(dir_path,'models','1_big_retry.pt')
		model = MyModel().from_pretrained(model_file)
		data_loader = model.preprocess([a.title for a in c],batch_size=4)
		embs = model.pred(data_loader,post_op='mean',cat=True)
		torch.save(embs,base+'embs.pt')
		subprocess.run('gsutil cp {} gs://keous-model-files/data/{}/'.format(base+'embs.pt',base),shell=True,check=False)


	def predict():
		from .models.bert_model import MyModel,read_out
		from .utils.article import Collection
		os.system('gsutil cp gs://keous-model-files/data/{}/{} .'.format(base,base+'collection.pkl'))
		c=Collection.load(base+'collection.pkl')
		model_file = os.path.join(dir_path,'models','3_new_data_model.pt')
		model = MyModel().from_pretrained(model_file,num_classes=5)
		x,_ = read_out(c,x_only=True,split=False,clean=False)
		data_loader = model.preprocess(x,batch_size=16)
		pred = model.pred(data_loader,post_op='predict',cat=True)
		c.load_predicted_sentiments(pred)
		c.save(base+'collection.pkl')
		os.system('gsutil cp {} gs://keous-model-files/data/{}/'.format(base+'collection.pkl',base))


	def pair():
		import torch
		from .utils.kb import build_dfs,KnowledgeBase
		from .utils.distance_calcs import get_pairs
		from .utils.article import Collection,Pairs
		from sklearn.cluster import OPTICS
		import numpy as np
		import spacy
		import neuralcoref
		nlp=spacy.load('en_core_web_lg')
		neuralcoref.add_to_pipe(nlp)

		os.system('gsutil cp gs://keous-model-files/data/{}/{} .'.format(base,base+'collection.pkl'))
		os.system('gsutil cp gs://keous-model-files/data/{}/{} .'.format(base,base+'embs.pt'))
		c=Collection.load(base+'collection.pkl')
		embs = torch.load(base+'embs.pt')
		cluster = OPTICS(min_samples=6).fit_predict(embs)
		kb=KnowledgeBase(nlp=nlp)
		df1,df2 = build_dfs(c,kb,df1_path=base+'sent_matrix.h5',df2_path=base+'mention_matrix.h5')
		kb.save(base+'kb.txt')
		pairs = get_pairs(cluster,df1=df1,df2=df2)
		article_pairs = [(sim,cos,np.linalg.norm(embs[i]-embs[j]),c[i],c[j]) if sim is not None else (None,None,None,None,None) for sim,cos,(i,j) in pairs]
		short_pairs = [p for p in article_pairs if p[0] is not None] #only relevent included
		paired = Pairs(short_pairs)

		with open(base+'pairs.json','w') as f:
		    json.dump(paired.to_json(),f)

		for p in article_pairs:
		        if p[0]==None:
		                print('No matches found')
		                print('\n')
		                continue
		        print(p[0])
		        print(p[1])
		        print(p[2])
		        print(p[3].title.strip(),p[3].source)
		        print(p[4].title.strip(),p[4].source)
		        print('\n')
		os.system('gsutil cp {} gs://keous-model-files/data/{}/'.format(base+'sent_matrix.h5',base))
		os.system('gsutil cp {} gs://keous-model-files/data/{}/'.format(base+'mention_matrix.h5',base))
		os.system('gsutil cp {} gs://keous-model-files/data/{}/'.format(base+'pair.pkl',base))


	commands = {
	'scrape_urls':scrape_urls,
	'build_collection':build_collection,
	'cluster':cluster,
	'predict':predict,
	'pair':pair,
	}

	if len(sys.argv)>1:
		command = sys.argv.pop(1)
		if command in commands:
			commands[command]()
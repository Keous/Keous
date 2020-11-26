import datetime
from .utils.scrape import scrape_all
from .utils.article import Collection
import os
import pytz
from itertools import chain
import spacy
import neuralcoref

nlp=spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)

base = datetime.datetime.now(pytz.utc).strftime('%m-%d-%y-')
c=scrape_all(max_n=25)
c.get_ents(nlp=nlp)
c.save(base+'collection.pkl')
#os.system('gsutil cp {} gs://keous-model-files/data/{}/'.format(base+'collection.pkl',base))
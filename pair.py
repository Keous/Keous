import os
import datetime
import torch
from .utils.kb import build_dfs,KnowledgeBase
from .utils.distance_calcs import get_pairs
from .utils.article import Collection,Pairs
from sklearn.cluster import OPTICS
import pytz
import pickle
import numpy as np

dir_path=os.path.dirname(os.path.realpath(__file__))
base = datetime.datetime.now(pytz.utc).strftime('%m-%d-%y-')
os.system('gsutil cp gs://keous-model-files/data/{}/{} .'.format(base,base+'collection.pkl'))
os.system('gsutil cp gs://keous-model-files/data/{}/{} .'.format(base,base+'embs.pt'))
c=Collection.load(base+'collection.pkl')
embs = torch.load(base+'embs.pt')
cluster = OPTICS(min_samples=2).fit_predict(embs) #6
kb=KnowledgeBase(base+'kb.txt')
df1,df2 = build_dfs(c,kb,df1_path=base+'sent_matrix.h5',df2_path=base+'mention_matrix.h5')
kb.save()
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
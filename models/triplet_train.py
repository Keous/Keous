from .bert_full import MyModel
from ..utils.article import Article,Collection
import os
os.system('gsutil cp gs://keous-model-files/data/{} .'.format('big_collection.msgpack'))
model = MyModel().fresh_load()
c=Collection().load('big_collection.msgpack')
model.triplet_train_collection(c,epochs=1,batch_size=3, headline_emb=True, save='paper_triplet_title_{}.pt')

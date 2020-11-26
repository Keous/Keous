from .bert_full import MyModel
from ..utils.article import Article,Collection
import pytz
import datetime
import os
base = datetime.datetime.now(pytz.utc).strftime('%m-%d-%y-')
#os.system('gsutil cp gs://keous-model-files/data/{}/{} .'.format(base,base+'collection.pkl'))
model = MyModel().fresh_load()
c=Collection.load('raw_articles.pkl')[1000:2000]+updater(base+'collection.pkl')
c=[a for a in c if hasattr(a,'title')]
print(len(c))
#model.triplet_train(c,epochs=1,n=3,save='{}_big_n=2')
model.triplet_train_collection(c,epochs=2,batch_size=3,save=base+'{}.pt')
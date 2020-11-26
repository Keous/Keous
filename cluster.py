import datetime
import os
from .models.bert_model import MyModel,read_out
from .utils.article import Collection
import torch
import pytz

dir_path=os.path.dirname(os.path.realpath(__file__))
base = datetime.datetime.now(pytz.utc).strftime('%m-%d-%y-')
os.system('gsutil cp gs://keous-model-files/data/{}/{} .'.format(base,base+'collection.pkl'))
c=Collection.load(base+'collection.pkl')
model_file = os.path.join(dir_path,'models','1_big_retry.pt')
model = MyModel().from_pretrained(model_file)
data_loader = model.preprocess([a.title for a in c],batch_size=4)
embs = model.pred(data_loader,post_op='mean',cat=True)
torch.save(embs,base+'embs.pt')
os.system('gsutil cp {} gs://keous-model-files/data/{}/'.format(base+'embs.pt',base))
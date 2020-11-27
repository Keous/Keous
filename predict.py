import datetime
import os
from .models.bert_model import MyModel,read_out
from .utils.article import Collection
import pytz

dir_path=os.path.dirname(os.path.realpath(__file__))
base = datetime.datetime.now(pytz.utc).strftime('%m-%d-%y-')
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
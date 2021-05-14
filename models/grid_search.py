from .models.bert_model import MyModel
from ..utils.read_data import read_sentihood
import os
import itertools
		
grid = {'lr': [2e-5], 'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}

def search(params, tr_batch_size=12, val_batch_size=16):
	lr, dropout = params
    model=MyModel().fresh_load(bert_files='new_transformers', num_classes=2, dropout_prob=dropout)
    
    xtr, ytr = read_sentihood('sentihood-train.json')
    xval, yval = read_sentihood('sentihood-dev.json')
    #tr_data_loader = model.preprocess(xtr, ytr, batch_size = tr_batch_size)
    #val_data_loader = model.preprocess(xval, yval, batch_size = val_batch_size)
    
    name = '{}'+'_sentihood_{}_{}.pt'.format(str(lr), str(dropout))
    model.supervised_train_data(xtr,ytr,xval,yval,batch_size=12,epochs=8,save=os.path.join('models',name), lr=lr, warmup=True)


for config in itertools.product(*grid.values()):
	print('Searching with {} and {} of {} and {}'.format(*grid.keys(), *config))
	search(config)

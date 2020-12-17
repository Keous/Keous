from .bert_full import MyModel,read_out
model=MyModel().fresh_load(num_classes=5,dropout_prob=0.5)
tr,val = read_out('out.pkl')
model.supervised_train_data(xtr,ytr,xval,yval,batch_size=4,epochs=1,save='{}_supervised_model.pt')
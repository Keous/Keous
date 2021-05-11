import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split

import pickle
import os
import time
import itertools
from collections import Counter
from tqdm import trange
import sys
import re
from itertools import chain

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import transformers
from transformers import BertModel, BertTokenizer, AdamW, BertPreTrainedModel, BertConfig, BertForSequenceClassification, BertModel

from ..utils.article import Article,Collection

def log(message,file='output.log'):
    with open(file,'a+') as f:
        f.write(message)
    print(message)

device = torch.device('cuda')

class MyBert(BertPreTrainedModel):
    def __init__(self,config,num_classes=None,dropout_prob=None):
        super().__init__(config)
        self.bert = BertModel(config)
        if num_classes is not None:
            self.cls = torch.nn.Linear(config.hidden_size,num_classes)
        if dropout_prob is not None:
            self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self,input_ids,attention_mask,post_op=None):
         last_hidden_states,pooled_output = self.bert(
                    input_ids,
                    attention_mask=attention_mask).to_tuple()
         if post_op=='mean': #meaned last hidden output (batch_size,768)
                return mean_pool(last_hidden_states,attention_mask)
         elif post_op=='default': #bert's pooling output (batch_size,768)
            return pooled_output
         elif post_op=='cls': #cls token (batch_size,768)
            return last_hidden_states[:,0,:]
         elif post_op==None: #last hidden output (batch_size,max_seq_len,768)
            return last_hidden_states
         elif post_op == 'predict':#make a prediction using linear layer, (batch_size,num_classes)
            if hasattr(self,'dropout'):
                pooled_output = self.dropout(pooled_output)
            logits = self.cls(pooled_output)
            return logits
         else:
            print('Invalid post_op. Must be one of: mean, default, cls, predict')





class MyModel(torch.nn.Module):
    def __init__(self,max_len=512,bert_files='HuggingFace Transformers'):
        super().__init__()
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer.model_max_length = 10**30 #to avoid annoying warnings

    def fresh_load(self,bert_files='HuggingFace Transformers',num_classes=None,dropout_prob=None):
        self.model = MyBert.from_pretrained(bert_files,num_classes=num_classes,dropout_prob=dropout_prob)
        self.model.cuda()
        return self

    def triplet_train(self,anchor,pos,neg,epochs=4,post_op='mean',save=None,lr=3e-5):
        self.setup_optimizer(lr=lr)
        log('Performing triplet training with {} epochs, {} post_op, and {} lr'.format(epochs,post_op,lr))
        criterion=torch.nn.TripletMarginLoss(margin=1.0, p=2)
        self.model.train()
        losses = []
        for e in trange(epochs):
            tr_loss=0
            tr_steps=0
            for (anchor_input_ids,anchor_mask,_),(pos_input_ids,pos_mask,_),(neg_input_ids,neg_mask,_) in zip(anchor,pos,neg):
                anchor_output = self.model(
                    anchor_input_ids.to(device),
                    anchor_mask.to(device),
                    post_op=post_op)
                pos_output = self.model(
                    pos_input_ids.to(device),
                    pos_mask.to(device),
                    post_op=post_op)
                neg_output = self.model(
                    neg_input_ids.to(device),
                    neg_mask.to(device),
                    post_op=post_op)

                self.model.zero_grad()
                
                loss = criterion(anchor_output,pos_output,neg_output)
                loss.backward()
                self.optimizer.step()
                tr_loss+= loss.item().  
                tr_steps += 1
            losses.append(tr_loss/tr_steps)
            log('Loss {}'.format(tr_loss/tr_steps))
            if save is not None:
                self.save(save.format(e))
                log('Saving to '+save.format(e))

    def supervised_train(self,train_data_loader,validation_data_loader,epochs=4,save=None,lr=3e-5,warmup=False):
        self.setup_optimizer(lr=lr,warmup=warmup,epochs=epochs,train_data_loader=train_data_loader)
        if hasattr(self.model,'dropout'):
            dropout = self.model.dropout.p
        else:
            dropout=None
        log('Performing supervised training with {}, {} dropout, {} lr, and {} warmup'.format(epochs,dropout,lr,warmup))
        criterion = torch.nn.CrossEntropyLoss()
        losses = []
        accuracy = []
        macro_f1 = []
        for e in trange(epochs):
            self.model.train()
            tr_loss=0
            tr_steps=0
            for input_id,mask,label in train_data_loader:
                logits = self.model(input_id.to(device),attention_mask=mask.to(device),post_op='predict')
                loss = criterion(logits,label.to(device))
                losses.append(loss.item())

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                if warmup == True:
                    self.scheduler.step()
                tr_loss += loss.item()
                tr_steps+=1
            log("Loss: {}".format(tr_loss/tr_steps))
            if save is not None:
                self.save(save.format(e))
                log('Saving file to '+save.format(e))
            self.model.eval()
            val_acc=0
            val_f1=0
            val_steps=0
            for input_id,mask,label in validation_data_loader:
                with torch.no_grad():
                    logits = self.model(input_id.to(device),attention_mask=mask.to(device),post_op='predict')
                y_pred = [np.argmax(logits).item() for logits in logits.to('cpu')]
                y_true = label
                acc=accuracy_score(y_true,y_pred)
                f1=f1_score(y_true,y_pred,average='macro')
                accuracy.append(acc)
                macro_f1.append(f1)
                val_acc+=acc
                val_f1+=f1
                val_steps+=1
            log('Acc {} and macro f1 {}'.format(val_acc/val_steps,val_f1/val_steps))
        return losses,accuracy,macro_f1

    def pred(self,data_loader,post_op='mean',cat=True):
        pred=[]
        self.model.eval()
        with torch.no_grad():
            for input_ids,attention_mask,_ in data_loader:
                output = self.model(
                    input_ids.to(device),
                    attention_mask.to(device),
                    post_op=post_op)
                if post_op=='predict':
                    pred.append([np.argmax(logit.to('cpu')).item() for logit in output]) #index of argmax is predicted class
                else:
                    pred.append(output.to('cpu'))

        if cat==False:
            return pred
        else:
            return np.concatenate(pred,axis=0)

    def triplet_train_collection(self,c,batch_size=8,epochs=4, headline_emb=True, post_op='default',save=None,lr=3e-5):
        if len(c)%2==1:
            c=c[:-1] #assure even number

        if headline_emb == True: #if  doing headline emb use double title
            anchor = [a.text() for a in c]
            pos = [a.title for a in c]
            neg = list(reversed([a.title for a in c]))

        if headline_emb == False: #if not doing headline emb use double text
            anchor = [a.title for a in c]
            pos = [a.text() for a in c]
            neg = list(reversed([a.text() for a in c]))

        anchor_loader = self.preprocess(anchor,batch_size=batch_size)
        pos_loader = self.preprocess(pos,batch_size=batch_size)
        neg_loader = self.preprocess(neg,batch_size=batch_size)
        self.triplet_train(anchor_loader,pos_loader,neg_loader,epochs=epochs,post_op=post_op,save=save,lr=lr)

    def supervised_train_data(self,xtr,ytr,xval,yval,batch_size=8,epochs=4,save=None,lr=3e-5,warmup=False):
        train_data_loader = self.preprocess(xtr,ytr,batch_size=batch_size)
        validation_data_loader = self.preprocess(xval,yval,batch_size=batch_size)
        return self.supervised_train(train_data_loader,validation_data_loader,epochs=epochs,save=save,lr=lr,warmup=warmup)


    def setup_optimizer(self,warmup=False,lr=3e-5,epochs=None,train_data_loader=None,warmup_percent=0.1):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}]
        if warmup == False:
            self.optimizer = AdamW(optimizer_grouped_parameters,lr=lr)
        elif warmup == True:
            self.optimizer = AdamW(optimizer_grouped_parameters,lr=lr)
            total_steps = len(train_data_loader)*epochs
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,num_warmup_steps=total_steps*warmup_percent,num_training_steps=total_steps)




    def preprocess(self,x,y=None,batch_size=32,shuffle=False):
        if y is not None:
            labels = torch.tensor(y).long()
        else:
            labels = torch.tensor([np.nan]*len(x))
        input_ids =[self.tokenizer.encode(text,add_special_tokens=True,padding=False,truncation=False,verbose=False) for text in x]
        input_ids = torch.tensor(pad_sequences(input_ids,maxlen=self.max_len)).to(torch.int64)
        attention_masks=[]
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
            
        masks = torch.tensor(attention_masks)
        data = TensorDataset(input_ids, masks, labels)
        if shuffle == False:
            dataloader = DataLoader(data, batch_size=batch_size)
        elif shuffle == True:
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, batch_size=batch_size, sampler=sampler)
        return dataloader


    def save(self,path):
        state = self.state_dict()
        torch.save(state,path)

    def load(self,path,strict=True,extra_args={}):
        state=torch.load(path)
        state.update(extra_args)
        self.load_state_dict(state,strict=strict)

    def from_pretrained(self,path,bert_files='HuggingFace Transformers',dropout_prob=None,num_classes=None,strict=True,extra_args={}):
        self.model = MyBert.from_pretrained(bert_files,dropout_prob=dropout_prob,num_classes=num_classes)
        self.load(path,strict=strict,extra_args=extra_args)
        self.model.cuda()
        return self



def pad_sequences(inputs,maxlen):
    padded=[]
    for item in inputs:
        if len(item)>maxlen:
            padded.append(item[:maxlen])
        elif len(item)<maxlen:
            padded.append(item+(maxlen-len(item))*[0]) #add 0's equal to the difference between maxlen and inputs
        else:
            padded.append(item)
    return padded

def mean_pool(token_embeddings,attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_emb = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    return sum_emb/sum_mask



def clean_out(out):
    new_out=[]
    for a in out:
        if a.title not in [art.title for art in new_out] and -100 not in a.ent_sents:
            #a.ent_sents = [e if e != -100 else 0 for e in a.ent_sents] #fill missing values (-100) with neutral/na (0)
            new_out.append(a)
    return new_out


def read_out(c,x_only=False,split=False,clean=True):
    if clean==True:    
        c=clean_out(c)
    x = []
    y = []
    for a in c:
        tot=0
        for para,ents in zip(a.paras,a.ents):
            text=para.strip()
            texts = [re.sub(re.escape(ent),'[TARGET]',text) for ent in ents]
            x.append(texts)

            if x_only == False:
                n=len(ents)
                labels = [s+2 for s in a.ent_sents[tot:n+tot]] #make all above 
                tot+=n
                y.append(labels)
    x=list(chain.from_iterable(x))
    y=list(chain.from_iterable(y))
    if split==True:
        xtr,xval,ytr,yval = train_test_split(x,y,test_size=0.2)
        return xtr,ytr,xval,yval
    elif split==False:
        return x,y
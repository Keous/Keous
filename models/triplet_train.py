from keous.models.bert_model import *
from keous.utils.article import Article,Collection
import os
model = MyModel().fresh_load()
self = model
c=Collection().load('big_collection.msgpack')[:200]
#model.triplet_train_collection(c,epochs=1,batch_size=3, headline_emb=True, save='paper_triplet_title_{}.pt')

batch_size=1
epochs=1
headline_emb=True
post_op='default'
save='paper_triplet_title_{}.pt'
lr=3e-5

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

anchor = self.preprocess(anchor,batch_size=batch_size)
pos = self.preprocess(pos,batch_size=batch_size)
neg = self.preprocess(neg,batch_size=batch_size)




self.setup_optimizer(lr=lr)
log('Performing triplet training with {} epochs, {} post_op, and {} lr'.format(epochs,post_op,lr))
criterion=torch.nn.TripletMarginLoss(margin=1.0, p=2)
self.model.train()
losses = []
tr_loss=0
tr_steps=0
for (anchor_input_ids,anchor_mask,_),(pos_input_ids,pos_mask,_),(neg_input_ids,neg_mask,_) in zip(anchor,pos,neg):
	pass

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
    tr_loss+= loss.item()
    tr_steps += 1
losses.append(tr_loss/tr_steps)
log('Loss {}'.format(tr_loss/tr_steps))
if save is not None:
    self.save(save.format(e))
    log('Saving to '+save.format(e))
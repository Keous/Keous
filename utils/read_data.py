import re
from itertools import chain
import json

def read_collection(c,x_only=False):
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
    return x,y

def read_sentihood(file):
    with open(file, 'r') as f:
        raw_text = f.read()

    x = []
    y = []

    data = json.loads(raw_text)
    for sentence in data:
        for point in sentence['opinions']:
            if point['sentiment'] != 'Neutral':
                x.append(sentence['text'].replace(point['target_entity'], '[TARGET]'))
                if point['sentiment'] == 'Positive':
                    y.append(1)
                elif point['sentiment'] == 'Negative':
                    y.append(0)
    return x,y


def read_uspol(file):
    with open(file, 'r') as f:
        data = f.read()
    data = json.loads(data)
    x = [item['text'] for item in data]
    y = [item['label'] for item in data]
    return x,y

from itertools import chain
import spacy
import neuralcoref
import random
from collections import Counter
import numpy as np
import pandas as pd
import operator
import re
import ast
from sklearn import metrics

nlp=spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)


class KB_Item():
    def __init__(self,iterable,kb,code=None):
        self.kb=kb
        if code is not None:
            self.code = code
        else:
            self.code=self.kb.generate_code()
        self.counter=Counter(iterable)
        self.update()

    def update(self):
        self.vector = self.get_vec()
        self.entries = [key for key in self.counter.keys()]


    def get_vec(self):
        vals=list(self.counter.values())
        total=sum(vals)
        weight_matrix=np.array(vals)/total
        vec_matrix=np.vstack([nlp(key).vector for key in self.counter.keys()])
        return np.dot(weight_matrix,vec_matrix)

    def __repr__(self):
        return str(dict(self.counter.items()))

    def add(self,items):
        new_c=Counter(items)
        self.counter+=new_c
        self.update()

    def sim(self,other):
        assert other.vector.shape == self.vector.shape
        if np.linalg.norm(self.vector)==0:
            return 0
        if np.linalg.norm(other.vector)==0:
            return 0
        return np.dot(self.vector,other.vector)/(np.linalg.norm(self.vector)*np.linalg.norm(other.vector))


class KnowledgeBase():
    def __init__(self,kb_file=None,seed_dict=None):
        if kb_file is None:
            return self

        self.kb_file=kb_file
        if seed_dict is not None:
            self.kb=seed_dict
        else:
            self.kb={}
            try:
                self.load()
            except FileNotFoundError:
                self.save()

    def save(self):
        with open(self.kb_file,'w',encoding='utf-8') as f:
            if len(self)==0:
                f.write('{}')
            for key,val in self.items():
                f.write('{}:{}\n'.format(key,val))

    def load(self,file=None):
        if file is None:
            file=self.kb_file
        with open(file,'r',encoding='utf-8') as f:
            data=f.readlines()
        if data==['{}']:
            self.kb =  {}
        else:
            for line in data:
                code=int(line[:13])
                body=ast.literal_eval(line[14:].strip())
                self.kb[code]=KB_Item(body,self,code=code)
            return self

    def generate_code(self):
        code=random.randint(10**12,10**13-1)
        if code in self.keys():
            code = self.generate_code()
        return code

    def add(self,iterable):
        new_item = KB_Item(iterable,kb=self)
        self.kb[new_item.code]=new_item
        return new_item

    def __repr__(self):
        return str(self.kb)

    def __getitem__(self,i):
        return self.kb[i]

    def __setitem__(self,i,item):
        self.kb[i]=item

    def __len__(self):
        return len(self.kb)

    def find_most_sim(self,queries):
        if len(self) == 0:
            return None
        sims = metrics.pairwise.cosine_similarity([s.vector for s in queries],[item.vector for item in self.values()])
        values = list(self.values())
        return [(values[idx],sims[i][idx]) for i,idx in enumerate(np.argmax(sims,axis=1))]

    def items(self):
        return self.kb.items()

    def values(self):
        return self.kb.values()

    def keys(self):
        return self.kb.keys()

    def load_cluster(self,a):
        doc=nlp(a.text())
        for cluster in doc._.coref_clusters:
            stop_free = [m for m in cluster.mentions if not (len(m)==1 and m[0].is_stop)] #rempove pronouns and other stops
            if len(stop_free)==0:
                continue
            else:
                most_sim = self.find_most_sim(stop_free)
                if most_sim is not None:
                    most_sim_dict = {}
                    for item in most_sim:
                        add_or_create(most_sim_dict,item[0],item[1])
                    best_fit,score = max(most_sim_dict.items(), key=operator.itemgetter(1))
                    if score[0]/score[1] > 0.85:
                        self[best_fit.code].add([m.text for m in stop_free])
                    else: self.add([m.text for m in stop_free])
                else:
                    self.add([m.text for m in stop_free])


    def build_ent_dict(self,a,use_code=False):
        annotations = a.ent_sents
        ents = list(chain.from_iterable(a.ents))
        docs = nlp.pipe(ents)
        most_sim = self.find_most_sim(docs)
        ent_dict={}
        for (best_fit,score),ann,ent in zip(most_sim,annotations,ents):
            if score>0.75:
                add_or_create(ent_dict,best_fit,ann,use_code=use_code)
            else:
                new_item = self.add([ent])
                add_or_create(ent_dict,new_item,ann,use_code=use_code)
        return ent_dict


def add_or_create(dictionary,key,val,use_code=False):
    if use_code==True:
        key=key.code
    if key in dictionary:
        dictionary[key][0]+=val
        dictionary[key][1]+=1
    else:
        dictionary[key]=[val,1]

def clean_out(out):
    new_out=[]
    for a in out:
        if a.title not in [art.title for art in new_out] and -100 not in a.ent_sents:
            new_out.append(a)
    return new_out

def append_many(df1,df2,ent_dicts):
    for ent_dict in ent_dicts:
        ent_dict_1 = {}
        ent_dict_2 = {}
        for key,val in ent_dict.items():
            ent_dict_1[key]=val[0]
            ent_dict_2[key]=val[1]
        df1=df1.append(ent_dict_1,ignore_index=True).fillna(2) #total sentiment
        df2=df2.append(ent_dict_2,ignore_index=True).fillna(0) #n mentions
    return df1,df2

def build_dfs(c,kb,df1_path='sent_matrix.h5',df2_path='mention_matrix.h5'):
    # c=clean_out(c)
    for a in c:
        kb.load_cluster(a)
    ent_dicts=[]
    for a in c:
        ent_dicts.append(kb.build_ent_dict(a,use_code=True))
    #try:
    #    df1=pd.read_hdf(df1_path,'fixed')
    #    df2=pd.read_hdf(df2_path,'fixed')
    #except FileNotFoundError:
    #    print('Making new files at {} and {}'.format(df1_path,df2_path))
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df1,df2=append_many(df1,df2,ent_dicts)
    df1.to_hdf(df1_path,'fixed')
    df2.to_hdf(df2_path,'fixed')
    return df1,df2
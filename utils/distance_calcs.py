import pandas as pd
import numpy as np
import torch
from itertools import combinations
from collections import Counter

def safe_divide(a,b):
	return np.divide(a, b, out=np.zeros_like(a)+2, where=b!=0)

def cos_by_idx(x,y,values):
	a=values[x]
	b=values[y]
	return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def sim_by_idx(x,y,values,weights):
	a=values[x]
	w_a=weights[x]
	b=values[y]
	w_b=weights[y]
	return f(a,w_a,b,w_b)

def f(a,w_a,b,w_b):
	'''Unit norm all input vectors, then multiply weights and values together and sum'''
	a = safe_divide(a,w_a) - 2
	b = safe_divide(b,w_b) - 2
	new=np.stack((a,w_a,b,w_b))
	norms = np.linalg.norm(new,axis=1)
	normed = new/norms[:,np.newaxis]
	a_norm,w_a_norm,b_norm,w_b_norm = norms
	return np.sum(w_a_norm*w_b_norm*np.abs(a-b))


def pairwise_sims(indices,values,weights,return_least_sim=True):
	sims=[]
	combs = combinations(indices,2)
	for i,j in combs:
		sims.append((sim_by_idx(i,j,values,weights),cos_by_idx(i,j,weights),(i,j)))
	if return_least_sim == False:
		return sims
	elif return_least_sim == True:
		ordered = sorted(sims,key=lambda x: x[0],reverse=True)
		for o in ordered:
			sim,cos,(a,b) = o
			if sim > 0.2 and cos>0.5: #cosine sim cutoff
				return o
		return None,None,(None,None)

def get_pairs(cluster,df1='sent_matrix.h5',df2='mention_matrix.h5'):
	if type(df1)==pd.DataFrame and type(df2)==pd.DataFrame:
		pass
	elif type(df1)==str and type(df2)==str:
		df1 = pd.read_hdf(df1,'fixed')
		df2 = pd.read_hdf(df2,'fixed')
	else:
		raise TypeError('Incompatable types {} and {} (should be both dataframe or both str-like paths)'.format(df1,df2))
	values = df1.values
	weights = df2.values
	cluster_indices = [np.where(cluster==i)[0] for i in range(len(Counter(cluster))-1)]
	pairs = [pairwise_sims(indices,values,weights) for indices in cluster_indices]
	return pairs
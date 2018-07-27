from random import uniform
import numpy as np
from collections import OrderedDict, defaultdict
from itertools import tee
import time

# -----------------------------------------------
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def compute_div_n(caps,n=1):
  aggr_div = []
  for k in caps:
      all_ngrams = set()
      lenT = 0.
      for c in caps[k]:
         tkns = c.split()
         lenT += len(tkns)
         ng = find_ngrams(tkns, n)
         all_ngrams.update(ng)
      aggr_div.append(float(len(all_ngrams))/ (1e-6 + float(lenT)))
  return np.array(aggr_div).mean(), np.array(aggr_div)

def compute_global_div_n(caps,n=1):
  aggr_div = []
  all_ngrams = set()
  lenT = 0.
  for k in caps:
      for c in caps[k]:
         tkns = c.split()
         lenT += len(tkns)
         ng = find_ngrams(tkns, n)
         all_ngrams.update(ng)
  if n == 1:
    aggr_div.append(float(len(all_ngrams)))
  else:
    aggr_div.append(float(len(all_ngrams))/ (1e-6 + float(lenT)))
  return aggr_div[0], np.repeat(np.array(aggr_div),len(caps))
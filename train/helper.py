import numpy as np
from time import time
import torch

def _sync():
    return 
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()

def all_keys(lis):
    if len(lis)==0:
        return set()
    keys=set()
    types={}
    for dic in lis:
        new_keys=set([key for key in dic.keys() if key not in keys])
        for key in new_keys:
            keys.add(key)
            types[key]=type(dic[key])
    return keys, types

def dic_mean(lis):
    if len(lis)==0:
        return {}
    keys, types=all_keys(lis)
    total={key:[] for key in keys}
    for dic in lis:
        for key in total:
            if key in dic:
                total[key].append(dic[key])
            else:
                total[key].append(None)
    for key in total:
        total[key]=np.array(total[key])
        #remove None
        total[key]=total[key][np.where(total[key]!=None)]
        #remove NaN
        try:
            if len(total[key])>0:total[key]=total[key][np.where(~np.isnan(total[key]))]
        except TypeError as e:
            print(f"Error in dic_mean for key {key}: {e}")
    return {key:(np.mean(total[key]) if len(total[key])>0 else -1.0) for key in total}
def dic_meanstd(lis):
    if len(lis)==0:
        return {}
    keys, types=all_keys(lis)
    total={key:[] for key in keys}
    for dic in lis:
        for key in total:
            if key in dic:
                total[key].append(dic[key])
            else:
                total[key].append(None)
    for key in total:
        total[key]=np.array(total[key])
        #remove None
        total[key]=total[key][np.where(total[key]!=None)]
        #remove NaN
        try:
            if len(total[key])>0:total[key]=total[key][np.where(~np.isnan(total[key]))]
        except TypeError as e:
            print(f"Error in dic_mean for key {key}: {e}")
    means={key:(np.mean(total[key]) if len(total[key])>0 else -1.0) for key in total}
    stds={key:(np.std(total[key])/np.sqrt(len(total[key])) if len(total[key])>0 else -1.0) for key in total}
    return {key:np.array([means[key], stds[key]]) for key in total}
    return means, stds

def dic_merge(lis):
    if len(lis)==0:
        return {}
    keys, types=all_keys(lis)
    total={key:[] for key in keys}
    for dic in lis:
        for key in total:
            if key in dic:
                total[key].append(dic[key])
            else:
                total[key].append(np.zeros(1, dtype=types[key])[0])
    #replace none with matching type
    return {key:np.array(total[key]) for key in total}

def dic_concat(lis):
    if len(lis)==0:
        return {}
    keys, types=all_keys(lis)
    total={key:[] for key in keys}
    for dic in lis:
        for key in total:
            if key in dic:
                total[key].append(dic[key])
            else:
                example_key=next(iter(dic))
                total[key].append(np.zeros(len(dic[example_key]), dtype=types[key]))
    return {key:np.concatenate(total[key]) for key in total}

totaltimedic={}
timedic={}

def time0(key):
    global timedic
    #torch.cuda.synchronize()
    _sync()
    timedic[key]=time()
def time1(key):
    global timedic, total_time_dic
    if key not in timedic:
        raise ValueError(f"Key {key} not found in timedic.")
    #torch.cuda.synchronize()
    _sync()
    elapsed = time() - timedic[key]
    if key not in totaltimedic:
        totaltimedic[key] = 0.0
    totaltimedic[key] += elapsed
    return elapsed
def totaltime(key):
    if key not in totaltimedic:
        raise ValueError(f"Key {key} not found in totaltimedic.")
    return totaltimedic[key]
def resettimes():
    global timedic, totaltimedic
    timedic={}
    totaltimedic={}
def alltimes(ident="times/"):
    return {ident+key:val for key, val in totaltimedic.items()}





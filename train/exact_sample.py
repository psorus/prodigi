import numpy as np

def _sample_exact_gauss(mean, diag_std, n_samples):
    if n_samples==0:
        return np.zeros((0, len(mean)), dtype=np.float32)
    dim=len(mean)
    full=np.random.normal(loc=mean, scale=diag_std, size=(int(n_samples*1.5), dim))
    ret=full[:n_samples]
    add=full[n_samples:]
    #I want to use the add samples to replace any samples so that the mean and diag_std is more correct
    current_mean=np.mean(ret, axis=0)
    current_std=np.std(ret, axis=0)
    current_loss=np.sum((current_mean-mean)**2)+np.sum((current_std-diag_std)**2)
    #print("before",current_loss)
    for i in range(len(add)):
        new_ret=np.copy(ret)
        new_ret[i%len(ret)]=add[i]
        new_mean=np.mean(new_ret, axis=0)
        new_std=np.std(new_ret, axis=0)
        new_loss=np.sum((new_mean-mean)**2)+np.sum((new_std-diag_std)**2)
        if new_loss<current_loss:
            ret=new_ret
            current_mean=new_mean
            current_std=new_std
            current_loss=new_loss
    return ret


def _sample_exact_gauss(mean, diag_std, n_samples):
    ret=np.random.normal(loc=mean, scale=diag_std, size=(n_samples, len(mean)))
    ret-=np.mean(ret, axis=0)
    ret/=(np.std(ret, axis=0)+1e-8)
    ret*=diag_std
    ret+=mean
    return ret

def _sample_exact_gauss_oned(mean, diag_std, n_samples):
    ret=_sample_exact_gauss(np.array([mean]), np.array([diag_std]), n_samples)
    ret=ret.reshape(-1)
    return ret

def _sample_exact_uniform(n_samples):
    #always assumes 0-1. I want the mean to 0.5 and the std to be 1/sqrt(12)
    if n_samples==0:
        return np.zeros((0, 1), dtype=np.float32)
    full=np.random.uniform(low=0.0, high=1.0, size=int(n_samples*1.5))
    ret=full[:n_samples]
    add=full[n_samples:]
    current_mean=np.mean(ret, axis=0)
    current_std=np.std(ret, axis=0)
    current_loss=np.sum((current_mean-0.5)**2)+np.sum((current_std-(1/np.sqrt(12)))**2)
    #print("before",current_loss)
    for i in range(len(add)):
        new_ret=np.copy(ret)
        new_ret[i%len(ret)]=add[i]
        new_mean=np.mean(new_ret, axis=0)
        new_std=np.std(new_ret, axis=0)
        new_loss=np.sum((new_mean-0.5)**2)+np.sum((new_std-(1/np.sqrt(12)))**2)
        if new_loss<current_loss:
            ret=new_ret
            current_mean=new_mean
            current_std=new_std
            current_loss=new_loss
    return ret
def _sample_exact_uniform(n_samples):
    ret=np.random.uniform(low=0.0, high=1.0, size=n_samples)
    ret-=np.mean(ret, axis=0)
    ret/=(np.std(ret, axis=0)+1e-8)
    ret*=1/np.sqrt(12)
    ret+=0.5
    #clip to 1e-8-1-1e-8
    ret=np.clip(ret, 1e-8, 1-1e-8)
    return ret



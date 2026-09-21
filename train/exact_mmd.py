import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from numba import njit



@njit(fastmath=True)
def linear_mmd_rbf(X, Y, gamma=1):
    n = min(X.shape[0], Y.shape[0])

    # need even number of samples
    n = n - (n % 2)

    mmd = 0.0

    for i in range(0, n, 2):

        # paired samples
        x1 = X[i]
        x2 = X[i + 1]

        y1 = Y[i]
        y2 = Y[i + 1]

        d_xx = 0.0
        d_yy = 0.0
        d_xy = 0.0
        d_yx = 0.0

        for k in range(X.shape[1]):

            t = x1[k] - x2[k]
            d_xx += t * t

            t = y1[k] - y2[k]
            d_yy += t * t
            t = x1[k] - y2[k]
            d_xy += t * t

            t = x2[k] - y1[k]
            d_yx += t * t

        k_xx = np.exp(-gamma * d_xx)
        k_yy = np.exp(-gamma * d_yy)
        k_xy = np.exp(-gamma * d_xy)
        k_yx = np.exp(-gamma * d_yx)

        mmd += k_xx + k_yy - k_xy - k_yx

    return mmd / (n // 2)


@njit(fastmath=True)
def quadratic_mmd_rbf(X, Y, gamma=1.0):
    n = X.shape[0]
    m = Y.shape[0]

    # XX term
    xx = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0
            for k in range(X.shape[1]):
                t = X[i, k] - X[j, k]
                d += t * t
            xx += np.exp(-gamma * d)

    xx *= 2.0 / (n * (n - 1))

    # YY term
    yy = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            d = 0.0
            for k in range(Y.shape[1]):
                t = Y[i, k] - Y[j, k]
                d += t * t
            yy += np.exp(-gamma * d)

    yy *= 2.0 / (m * (m - 1))

    # XY term
    xy = 0.0
    for i in range(n):
        for j in range(m):
            d = 0.0
            for k in range(X.shape[1]):
                t = X[i, k] - Y[j, k]
                d += t * t
            xy += np.exp(-gamma * d)

    xy /= (n * m)

    return xx + yy - 2.0 * xy

def sample_mmd(samples1, samples2, dim, gamma=1.0):
    return quadratic_mmd_rbf(samples1, samples2, gamma/dim)

def just_mmd(prior,gmm1,gmm2, num_samples=1000, gamma=1.0):
    samples1 = prior.sample(gmm1,num_samples)
    samples2 = prior.sample(gmm2,num_samples)

    #mmd=linear_mmd_rbf(samples1, samples2, gamma)
    dim=samples1.shape[1]
    mmd=quadratic_mmd_rbf(samples1, samples2, gamma/dim)

    return mmd

def metrics(prior, gmm1, gmm2, num_samples=1000, tries=100, gamma=1.0):
    samples1=prior.sample(gmm1, num_samples)
    samples2=prior.sample(gmm2, num_samples)
    dim=len(samples1[0])
    mmd=quadratic_mmd_rbf(samples1, samples2, gamma/dim)

    nulls=[quadratic_mmd_rbf(samples1, prior.sample(gmm1, num_samples), gamma/dim) for _ in range(tries)]
    nulls=np.array(nulls)
    avg_null=np.mean(nulls)
    sigma_null=np.std(nulls)+1e-8
    deltaOsigma=(mmd-avg_null)/sigma_null
    return {"mmd":mmd, "pvalue":float(np.mean(mmd>nulls)), "deltaOsigma":float(deltaOsigma)}

def fast_metrics(prior,gmm1, gmm2, num_samples=1000, tries=100, gamma=1.0):
    samples1 = prior.sample(gmm1,num_samples)
    samples2 = prior.sample(gmm2,num_samples)

    #mmd=linear_mmd_rbf(samples1, samples2, gamma)
    mmd=quadratic_mmd_rbf(samples1, samples2, gamma)

    nulls=[quadratic_mmd_rbf(prior.sample(gmm1,num_samples), prior.sample(gmm1,num_samples), gamma) for _ in range(tries)]
    average_null=np.mean(nulls)
    sigma_null=np.std(nulls)
    pvalue=np.mean(np.array(nulls)>=mmd)

    deltaOsigma=np.abs(mmd-average_null)/sigma_null if sigma_null>0 else -1.0
    if deltaOsigma>100:
        deltaOsigma=100.0

    rejectNull=float(int(pvalue<0.05))
    return {"recon/mmd":mmd,"recon/pvalue":pvalue,"recon/deltaOsigma":deltaOsigma, "recon/average_null":average_null, "recon/sigma_null":sigma_null, "recon/rejectNull":rejectNull}




def rbf_kernel(X, Y, gamma):
    sq_dists = cdist(X, Y, metric='sqeuclidean')
    return np.exp(-gamma * sq_dists)


def mmd_rbf(X, Y, gamma=1.0):
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)

    n = X.shape[0]
    m = Y.shape[0]

    # Remove diagonal terms
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    term_xx = Kxx.sum() / (n * (n - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.sum() / (n * m)

    return term_xx + term_yy - 2 * term_xy


def mmd_distance(gmm1, gmm2, num_samples=250, gamma=1.0):
    samples1 = gmm1.sample(num_samples)
    samples2 = gmm2.sample(num_samples)

    return mmd_rbf(samples1, samples2, gamma)

def mmd_pvalue(gmmExp, gmmNew, tries=250, num_samples=100, gamma=1.0):
    direct=mmd_distance(gmmExp, gmmNew, num_samples, gamma)
    nulls=[]
    for i in range(tries):#tqdm(range(tries), "Calculating pvalue"):
        nulls.append(mmd_distance(gmmExp,gmmExp,num_samples,gamma))
    nulls=np.array(nulls)
    pvalue=np.mean(nulls>=direct)
    return pvalue


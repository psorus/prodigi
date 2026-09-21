from prior import Prior
import numpy as np
from scipy.stats import norm, t, beta, expon, uniform
from copula_prior import CopulaPrior
from gmm_prior import GMMPrior
from scm_prior import SCMPrior


class MixedPrior(Prior):
    def __init__(self,cfg=None, pattern="mixed_pattern.boso",priors=None):
        super().__init__(cfg,pattern)
        if priors is None:
            priors={"gmm":GMMPrior(self.cfg), "copula":CopulaPrior(self.cfg), "scm":SCMPrior(self.cfg)}
        self.priors=priors
    def allowed_args(self, cfg=None)->list:
        lis=[]
        for family,prior in self.priors.items():
            for arg in prior.allowed_args(cfg):
                curr={"family":family}
                curr.update(arg)
                lis.append(curr)
            
        return lis


    def random_distribution(self, family="gmm",**kwargs):
        curr=self.priors[family].random_distribution(**kwargs)
        curr["family"]=family
        return curr


    def sample(self, dic, n_samples=1000)->np.array:
        family=dic["family"]
        return self.priors[family].sample(dic, n_samples=n_samples)
    
    def sample_exact(self, dic, n_samples=1000)->np.array:
        family=dic["family"]
        return self.priors[family].sample_exact(dic, n_samples=n_samples)
    def sample_torch(self, dic, n_samples=1000):
        family=dic["family"]
        return self.priors[family].sample_torch(dic, n_samples=n_samples)


    def template_identifier(self, dic)->dict:
        ret={"family":dic["family"]}
        ret.update(self.priors[dic["family"]].template_identifier(dic))
        return ret
    
    def template_variables(self, dic)->dict:
        ret={"family":dic["family"]}
        ret.update(self.priors[dic["family"]].template_variables(dic))
        return ret

    def meta_info(self, dic)->dict:
        ret={"family":dic["family"]}
        ret.update(self.priors[dic["family"]].meta_info(dic))
        return ret

    def posthoc(self, dic)->bool:
        family=dic["family"]
        return self.priors[family].posthoc(dic)

    def posthoc_debug(self, dic)->bool:
        family=dic["family"]
        return self.priors[family].posthoc_debug(dic)

    def normalize(self, dic, method="zscore"):
        family=dic["family"]
        return self.priors[family].normalize(dic, method=method)
    def denormalize(self, dic, norm_params, method="zscore"):
        family=dic["family"]
        return self.priors[family].denormalize(dic, norm_params, method=method)

    def density(self, dic, samples, eps=1e-6)->np.array:
        family=dic["family"]
        return self.priors[family].density(dic, samples,    )

    def score(self, dic, samples, eps=1e-6)->np.array:
        family=dic["family"]
        return self.priors[family].score(dic, samples, eps=eps)


    def prep_optimization(self, dic):
        family=dic["family"]
        ret, var=self.priors[family].prep_optimization(dic)
        ret["family"]=family
        return ret, var

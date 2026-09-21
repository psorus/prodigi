from boso import BElem, BMatchingError
import numpy as np
import torch

class Prior():
    def __init__(self, cfg=None, template="train/gmm_pattern.boso"):
        if cfg is None:
            from config import Config
            cfg=Config()
        self.template_path=cfg.TEMPLATE_PATH_BEFORE+template+cfg.TEMPLATE_PATH_AFTER
        with open(self.template_path, 'r') as f:
            self.template_string = f.read()
        self.boso = BElem.parse(self.template_string)
        self.cfg=cfg

    def arg_to_str(self,arg):
        family=arg["family"] if "family" in arg else self.template_path.split("/")[-1].split("_")[0]
        arg={key:value for key,value in arg.items() if key!="family"}
        #sort by key
        arg=dict(sorted(arg.items(), key=lambda x:x[0]))
        stri=family+"-".join([str(zw) for zw in arg.values()])
        return stri

    def random_distribution(self, *args, **kwargs)->dict:
        raise NotImplementedError(f"Method random_distribution() not implemented in {self.__class__.__name__}")

    def allowed_args(self, cfg=None)->list:
        raise NotImplementedError(f"Method allowed_args() not implemented in {self.__class__.__name__}")

    def sample(self, dic, n_samples=1000)->np.array:
        raise NotImplementedError(f"Method sample() not implemented in {self.__class__.__name__}")

    def sample_exact(self, dic, n_samples=1000)->np.array:
        raise NotImplementedError(f"Method sample_exact() not implemented in {self.__class__.__name__}")

    def sample_torch(self, dic, n_samples=1000)->torch.Tensor:
        return torch.from_numpy(self.sample(dic, n_samples)).float()

    def template_identifier(self, dic)->dict:
        raise NotImplementedError(f"Method template_identifier() not implemented in {self.__class__.__name__}")
    def identifying_tuple(self, dic)->tuple:
        dic=self.template_identifier(dic)
        return tuple(dic[key] for key in sorted(dic.keys()))

    def meta_info(self, dic)->dict:
        raise NotImplementedError(f"Method meta_info() not implemented in {self.__class__.__name__}")

    def dim(self, dic)->int:
        return self.template_identifier(dic)['dim']

    def describe(self, dic)->[]:
        return self.boso.manifest(dic)

    def template_variables(self, dic)->dict:
        return self.template_identifier(dic)

    def template(self, dic)->[]:
        return self.boso.template(self.template_variables(dic))

    def match(self, tokens)->dict:
        dic= self.boso.match(tokens)
        if not self.posthoc(dic):
            raise BMatchingError(f"Posthoc check failed for {dic}")
        return dic

    def match_debug(self, tokens)->dict:
        dic= self.boso.match(tokens)
        self.posthoc_debug(dic)
        return dic

    def posthoc(self, dic)->bool:
        return True

    def posthoc_debug(self, dic)->bool:
        raise NotImplementedError(f"Method posthoc_debug() not implemented in {self.__class__.__name__}")

    def normalize(self, dic, method="zscore"):
        #takes a prior description (dic), algorithmically normalizes it according to method, and returns both the normalized prior and a dictionary of normalization parameters used
        raise NotImplementedError(f"Method normalize() not implemented in {self.__class__.__name__}")

    def denormalize(self, dic, norm_params, method="zscore")->dict:
        #takes a normalized prior description (dic) and a dictionary of normalization parameters used, and returns the denormalized prior
        raise NotImplementedError(f"Method denormalize() not implemented in {self.__class__.__name__}")

    def density(self, dic, samples, eps=1e-6)->np.array:
        #takes a prior description (dic) and a set of samples, and returns the log density of the samples under the prior
        raise NotImplementedError(f"Method density() not implemented in {self.__class__.__name__}")

    def score(self, dic, samples, eps=1e-6)->np.array:
        #takes a prior description (dic) and a set of samples, and returns the log density of the samples under the prior
        raise NotImplementedError(f"Method score() not implemented in {self.__class__.__name__}")

    def prep_optimization(self, dic):
        #takes a prior description (numpy format), and converts each parameter that can be optimized (sample_torch) to a gradient torch tensor. Also returns the list of optimizable parameters
        raise NotImplementedError(f"Method prep_optimization() not implemented in {self.__class__.__name__}")


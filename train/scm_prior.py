from prior import Prior
import numpy as np
import hashlib
import scipy
from scipy.stats import norm, t, beta, expon, uniform
from boso import BMatchingError
from errors import SCMError, IncalculableError
from helper import time0, time1

from exact_sample import _sample_exact_gauss_oned, _sample_exact_uniform

from structural_generator import generate_structure, find_max_max_parents

import torch


funcs = {
    "tanh":        np.tanh,
    #"leakyrelu":   lambda x: np.where(x > 0, x, 0.01 * x),
    #"elu":         lambda x: np.where(x > 0, x, np.exp(x) - 1),
    #"identity":    lambda x: x,
    #"sigmoid":     lambda x: 1 / (1 + np.exp(-x)),
    #"softsign":    lambda x: x / (1 + np.abs(x)),
    #"hardtanh":    lambda x: np.clip(x, -1, 1),
    #"relu":        lambda x: np.maximum(0, x),
    #"softplus":    lambda x: np.log1p(np.exp(x)),
    #"selu":        lambda x: np.where(x > 0, 1.0507 * x, 1.0507 * 1.6733 * (np.exp(x) - 1)),
    #"sin":         np.sin,
    #"swish":       lambda x: x / (1 + np.exp(-x)),
    #"mish":        lambda x: x * np.tanh(np.log1p(np.exp(x))),
    #"gelu":        lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
    #"lognormal": lambda u, mu=0, sigma=1: np.exp(mu + sigma * norm.ppf(u)),
    #"normal":    lambda u, mu=0, sigma=1: mu + sigma * norm.ppf(u),
    #"laplace":   lambda u, mu=0, sigma=1: mu - sigma * np.sign(u - 0.5) * np.log(1 - 2*np.abs(u - 0.5)),
    #"cauchy":    lambda u, mu=0, sigma=1: mu + sigma * np.tan(np.pi * (u - 0.5)),
    #"logistic":  lambda u, mu=0, sigma=1: mu + sigma * np.log(u / (1 - u)),
    #"arctan":   lambda x: np.arctan(x),
    #"erf":      lambda x: scipy.special.erf(x),     # from scipy.special
    #"sinh":     lambda x: np.sinh(x),
    #"asinh":        lambda x: np.arcsinh(x),                          # = ln(x + sqrt(x^2+1))
    #"bent_identity": lambda x: (np.sqrt(x**2 + 1) - 1) / 2,
    #"square": lambda x: x**2,
    #"root": lambda x: np.sqrt(np.abs(x)),
    #"log": lambda x: np.log(np.abs(x) + 1),  # Added small constant to avoid log(0)
    "oddbump": lambda x: (x)/(1+x*x),
    "bump": lambda x: 1/(1+x*x),#np.exp(-x**2),
    #"asym": lambda x: x*np.exp(-0.5*x*x),
}

#allowed_funcs=["tanh","sigmoid","softplus","elu","gelu","mish","sin","softsign", "arctan","asinh","bent_identity"]
allowed_funcs=["tanh","oddbump", "bump"]

init_funcs=allowed_funcs#["tanh","sigmoid","relu","softplus"]
parent_funcs=allowed_funcs#["tanh","elu","leakyrelu","sigmoid","relu","softplus"]
noise_funcs=init_funcs

def random_func(layer):
    if layer==0:
        return np.random.choice(init_funcs)
    if layer<0:
        return np.random.choice(noise_funcs)
    return np.random.choice(parent_funcs)


class SCMPrior(Prior):
    def __init__(self,cfg=None):
        super().__init__(cfg,"scm_pattern.boso")
        self.depth_mode=self.cfg.DEPTH_MODE.lower()
        self.depth_min=self.cfg.DEPTH_MIN
        self.depth_max=self.cfg.DEPTH_MAX

    def allowed_args(self, cfg=None)->list:
        if cfg is None:cfg=self.cfg
        MIN_DIM, MAX_DIM=cfg.MIN_DIM, cfg.MAX_DIM
        MIN_MAX_PARENTS=cfg.MIN_MAX_PARENTS
        MAX_MAX_PARENTS=cfg.MAX_MAX_PARENTS
        MAX_ALLOWED_HIDDEN=cfg.MAX_ALLOWED_HIDDEN
        lis=[]

        for dim in range(MIN_DIM, MAX_DIM+1):
            if cfg.SIMPLE_MAX_PARENTS:
                MAX_MAX_PARENTS=min(cfg.MAX_MAX_PARENTS, dim-1)
            else:
                MAX_MAX_PARENTS=find_max_max_parents(dim, cfg.MAX_MAX_PARENTS)
            for max_parents in range(min(MIN_MAX_PARENTS,MAX_MAX_PARENTS), MAX_MAX_PARENTS+1):
                for hidden_nodes in range(0, MAX_ALLOWED_HIDDEN+1):
                    lis.append({"dim":dim, "max_parents":max_parents, "hidden_nodes":hidden_nodes})

        return lis

    def random_distribution(self, dim=5, max_parents=2, hidden_nodes=0,  **kwargs):
        depth=None
        if self.depth_mode=="auto":
            depth=None
        elif self.depth_mode=="uniform":
            depth=np.random.randint(self.depth_min, self.depth_max+1)
        structure=generate_structure(dim, max_parents, hidden_nodes, maximum_depth=depth)
        tdim=dim+hidden_nodes
        function=[]
        parents=[]
        factor=[]
        observed=[]
        parent_count=[]
        
        for dic in structure:
            index, par, typ=dic["id"],dic["parents"], dic["type"]
            if typ=="zoey":
                func=random_func(-1)
            elif len(par)==0:
                func=random_func(0)
            else:
                func=random_func(1)
            fact=np.zeros(max_parents, dtype=float)
            fact[:len(par)]=np.random.uniform(0.5,1.5,size=len(par))*np.random.choice([-1,1], size=len(par))
            parent_count.append(len(par))
            while len(par)<max_parents:
                par.append(-1)

            function.append(func)
            parents.append(np.array(par, dtype=int))
            factor.append(fact)
            observed.append(typ!="hidden")
                

        function=np.array(function)
        parents=np.array(parents)
        parent_count=np.array(parent_count)
        factor=np.array(factor)
        observed=np.array(observed)
        noises=np.exp(np.random.uniform(np.log(0.1),np.log(1.0), size=tdim))
        bias_before=np.random.normal(0,1,size=tdim)
        bias_after=np.random.normal(0,1,size=tdim)
        scales=np.random.uniform(0.5,1.5,size=tdim)

        dist={"dim":dim,
              "tdim":tdim,
              "hidden_nodes":hidden_nodes,
              "max_parents":max_parents,
              "function":function,
              "parents":parents,
              "parent_count":parent_count,
              "factor":factor,
              "noises":noises,
              "observed":observed,
              "bias_before":bias_before,
              "bias_after":bias_after,
              "scales":scales,
              }

        return dist

    def _run(self, dic, n_samples, source_sampler, noise_sampler)->tuple:
        dim, tdim, function,  noises, observed = dic["dim"], dic["tdim"], dic["function"], dic["noises"], dic["observed"]
        if "parents" in dic and "factor" in dic:
            parents, factor = dic["parents"], dic["factor"]
        else:
            parents=[[]]
            factor=[[]]
        
        bias_before=dic["bias_before"]
        bias_after=dic["bias_after"]
        scales=dic["scales"]

        observed=np.zeros(tdim, dtype=bool)
        observed[:dim]=True
        observed_indice=np.array([i for i in range(tdim) if observed[i]])

        ret=np.zeros((n_samples, tdim))
        available={int(i):False for i in range(tdim)}

        may_skip=False
        while True:
            did_something=False
            for index,avail in available.items():
                if avail:continue
                pars=parents[index]
                pars=[p for p in pars if p>=0]
                if may_skip or all(int(p) in available and available[int(p)] for p in pars):
                    func=function[index]
                    fac=factor[index]
                    funcname=str(func).lower()
                    if not funcname in funcs:
                        raise BMatchingError(f"Function {funcname} is not defined in the function dictionary.")
                    pos=index
                    if len(pars)==0 or may_skip:
                        raw=funcs[funcname](source_sampler(n_samples)+bias_before[pos])*scales[pos]+bias_after[pos]
                    else:
                        raw=(funcs[funcname](np.sum(ret[:,[p for p in pars]]*fac[:len(pars)], axis=1)+bias_before[pos])+noise_sampler(n_samples)*noises[pos])*scales[pos]+bias_after[pos]
                    ret[:,pos]=raw

                    available[index]=True
                    did_something=True
                    may_skip=False

            if not did_something:
                may_skip=True
            else:
                may_skip=False
            if all(list(available.values())):
                break

        return ret[:,observed_indice]


    def sample(self, dic, n_samples=1000)->np.array:
        out=self._run(
            dic, n_samples,
            source_sampler=lambda n: np.random.normal(0,1,size=n),
            #noise_sampler=lambda n: np.random.uniform(size=n),
            noise_sampler=lambda n: np.random.normal(0,1,size=n),
        )
        return out

    def sample_exact(self, dic, n_samples=1000)->np.array:
        out=self._run(
            dic, n_samples,
            source_sampler=lambda n: _sample_exact_gauss_oned(0,1,n),
            #noise_sampler=lambda n: _sample_exact_uniform(n),
            noise_sampler=lambda n:  _sample_exact_gauss_oned(0,1,n),
        )
        return out

    def sample_torch(self, dic, n_samples=1000):
        #currently 100% chatgpt
        """
        Differentiable PyTorch version of sample().

        Randomness is reparameterized using torch.randn, so gradients flow
        through:
            factor
            noises
            bias_before
            bias_after
            scales

        The discrete SCM structure (parents, functions, observed variables)
        remains fixed/non-differentiable, just as in sample().
        """

        dim = dic["dim"]
        tdim = dic["tdim"]
        function = dic["function"]
        parents = dic["parents"]

        # ------------------------------------------------------------------
        # Convert differentiable parameters to torch tensors.
        # ------------------------------------------------------------------
        device = None

        for key in ["factor", "noises", "bias_before", "bias_after", "scales"]:
            if isinstance(dic[key], torch.Tensor):
                device = dic[key].device
                break

        if device is None:
            device = torch.device("cpu")

        def as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(x, dtype=torch.float32, device=device)

        factor = as_tensor(dic["factor"])
        noises = as_tensor(dic["noises"])
        bias_before = as_tensor(dic["bias_before"])
        bias_after = as_tensor(dic["bias_after"])
        scales = as_tensor(dic["scales"])

        # ------------------------------------------------------------------
        # Torch versions of the SCM nonlinearities.
        #
        # Do NOT use the numpy funcs dictionary here: converting a tensor to
        # numpy would destroy the autograd graph.
        # ------------------------------------------------------------------
        torch_funcs = {
            "tanh": torch.tanh,
            "oddbump": lambda x: x / (1.0 + x * x),
            "bump": lambda x: 1.0 / (1.0 + x * x),
        }

        # ------------------------------------------------------------------
        # Same semantics as _run():
        #
        # source_sampler ~ N(0,1)
        # noise_sampler  ~ N(0,1)
        # ------------------------------------------------------------------
        source = torch.randn(
            n_samples,
            device=device,
            dtype=bias_before.dtype,
        )

        # Keep individual node outputs as tensors rather than doing
        # ret[:, pos] = raw. This gives a clean autograd graph.
        ret = [None] * tdim

        available = {int(i): False for i in range(tdim)}

        may_skip = False

        while True:
            did_something = False

            for index, avail in available.items():
                if avail:
                    continue

                pars = parents[index]
                pars = [int(p) for p in pars if p >= 0]

                # Same scheduling condition as _run().
                if may_skip or all(
                    p in available and available[p]
                    for p in pars
                ):
                    funcname = str(function[index]).lower()

                    if funcname not in torch_funcs:
                        raise BMatchingError(
                            f"Function {funcname} is not defined "
                            f"in the torch function dictionary."
                        )

                    func = torch_funcs[funcname]

                    # ------------------------------------------------------
                    # Root node / skipped node
                    # ------------------------------------------------------
                    if len(pars) == 0 or may_skip:
                        raw = (
                            func(source + bias_before[index])
                            * scales[index]
                            + bias_after[index]
                        )

                    # ------------------------------------------------------
                    # Normal structural equation
                    # ------------------------------------------------------
                    else:
                        parent_values = torch.stack(
                            [ret[p] for p in pars],
                            dim=1,
                        )

                        parent_factors = factor[index, :len(pars)]

                        parent_input = torch.sum(
                            parent_values * parent_factors.unsqueeze(0),
                            dim=1,
                        )

                        raw = (
                            func(parent_input + bias_before[index])
                            + torch.randn(
                                n_samples,
                                device=device,
                                dtype=bias_before.dtype,
                            ) * noises[index]
                        )

                        raw = (
                            raw * scales[index]
                            + bias_after[index]
                        )

                    ret[index] = raw
                    available[index] = True
                    did_something = True
                    may_skip = False

            if not did_something:
                may_skip = True
            else:
                may_skip = False

            if all(available.values()):
                break

        # Same observed-node semantics as sample():
        # observed = first `dim` variables.
        return torch.stack(ret[:dim], dim=1)



    def template_identifier(self, dic)->dict:
        return {"dim":dic["dim"],
                "max_parents":dic["max_parents"],
                "hidden_nodes":dic["hidden_nodes"],
                }
    def template_variables(self, dic)->dict:
        return {"dim":dic["dim"],
                "tdim":dic["tdim"],
                "max_parents":dic["max_parents"],
                "hidden_nodes":dic["hidden_nodes"],
                }
    def meta_info(self, dic)->dict:
        return self.template_identifier(dic)

    def posthoc(self, dic):
        for func in dic["function"]:
            if not str(func).lower() in funcs:
                return False
        return True


    def posthoc_debug(self, dic):
        samples=self.sample(dic, n_samples=1)

    def normalize(self, dic, method="zscore", n_samples_estimate=10_000)->tuple:
        if method=="zscore":
            tdim=dic["tdim"]
            bias_before=np.copy(dic["bias_before"])
            bias_after=np.copy(dic["bias_after"])
            parents=np.copy(dic["parents"])
            factor=np.copy(dic["factor"])
            noises=np.copy(dic["noises"])
            scales=np.copy(dic["scales"])

            samples=self.sample_exact(dic, n_samples=n_samples_estimate)
            scale_factor=np.std(samples, axis=0)
            scale_factor=np.clip(scale_factor, 1e-4, None)
            bias_delta=np.mean(samples, axis=0)

            available={int(i):False for i in range(tdim)}

            may_skip=False
            did_something=False
            while True:
                did_something=False
                for index,avail in available.items():
                    if avail:continue
                    pars = parents[index]
                    pars = [p for p in pars if p >=0] 

                    if may_skip or all(int(p) in available and available[int(p)] for p in pars):
                        bias_after[index] = (bias_after[index] - bias_delta[index]) / scale_factor[index]
                        scales[index] /= scale_factor[index]

                        bias_before_delta = 0.0
                        for parent_index, p in enumerate(pars):
                            bias_before_delta += bias_delta[p] * factor[index, parent_index]
                            factor[index, parent_index] *= scale_factor[p]
                        bias_before[index] += bias_before_delta

                        available[index] = True
                        did_something = True
                        may_skip = False

                if not did_something:
                    may_skip=True
                else:
                    may_skip=False
                if all(list(available.values())):
                    break

            ret={key:val for key,val in dic.items() if key not in ["bias_before", "bias_after", "scales", "factor"]}
            ret["bias_before"]=bias_before
            ret["bias_after"]=bias_after
            ret["scales"]=scales
            ret["factor"]=factor
            ret["parents"]=parents
            norm_info={"mean":bias_delta, "std":scale_factor}
        else:
            raise NotImplementedError
        return ret, norm_info

    def denormalize(self, dic, norm_info, method="zscore")->dict:
        if method=="zscore":
            tdim=dic["tdim"]
            bias_before=np.copy(dic["bias_before"])
            bias_after=np.copy(dic["bias_after"])
            parents=np.copy(dic["parents"])
            factor=np.copy(dic["factor"])
            scales=np.copy(dic["scales"])

            mean=norm_info["mean"]
            std=norm_info["std"]

            for index in range(tdim):
                pars = parents[index]
                pars = [p for p in pars if p >= 0]

                scales[index] *= std[index]

                bias_after[index] = bias_after[index] * std[index] + mean[index]

                bias_before_delta = 0.0
                for parent_index, p in enumerate(pars):
                    factor[index, parent_index] /= std[p]
                    bias_before_delta += mean[p] * factor[index, parent_index]
                bias_before[index] -= bias_before_delta

            ret={key:val for key,val in dic.items() if key not in ["bias_before", "bias_after", "scales", "factor"]}
            ret["bias_before"]=bias_before
            ret["bias_after"]=bias_after
            ret["scales"]=scales
            ret["factor"]=factor
            ret["parents"]=parents
        else:
            raise NotImplementedError
        return ret

    def density(self, dic, samples, eps=1e-6)->np.array:
        raise IncalculableError()

    def score(self, dic, samples, eps=1e-6)->np.array:
        raise IncalculableError()

    def prep_optimization(self, dic):
        keys=["factor","noises","bias_before","bias_after","scales"]
        dic_torch={key:val for key,val in dic.items()}
        for key in keys:
            dic_torch[key]=torch.tensor(dic[key].copy(), dtype=torch.float32, requires_grad=True)
        variables=[dic_torch[key] for key in keys]
        return dic_torch, variables



if __name__=="__main__":
    from config import Config
    cfg=Config()
    prior=SCMPrior(cfg)
    dist=prior.random_distribution(dim=5, max_parents=4, hidden_nodes=0)
    print(dist)
    dist2=prior.match(prior.describe(dist))
    print(dist2)
    from plt import plt
    samp=prior.sample(dist, n_samples=1000)
    plt.scatter(samp[:,0], samp[:,1])
    plt.show()
    for token in prior.describe(dist):
        print(token)
    print(dist["parents"])

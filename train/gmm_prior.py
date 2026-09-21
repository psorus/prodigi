from prior import Prior
import numpy as np
from helper import time0, time1
import torch

from scipy.special import logsumexp


from exact_sample import _sample_exact_gauss
from weight_generator import generate_weights


class GMMPrior(Prior):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg, template="gmm_pattern.boso")
        self.minimum_weight=self.cfg.MINIMUM_WEIGHT
        self.minimum_delta=self.cfg.MINIMUM_DELTA
    def allowed_args(self, cfg=None)->list:
        if cfg is None:cfg=self.cfg
        MIN_DIM, MAX_DIM, MIN_COMPONENTS, MAX_COMPONENTS=cfg.MIN_DIM, cfg.MAX_DIM, cfg.MIN_COMPONENTS, cfg.MAX_COMPONENTS
        lis=[]
        for dim in range(MIN_DIM, MAX_DIM+1):
            for ncomp in range(MIN_COMPONENTS, MAX_COMPONENTS+1):
                lis.append({"dim":dim, "ncomp":ncomp})
        return lis

    def random_distribution(self, dim=2, ncomp=3, **kwargs):
        weights=generate_weights(ncomp,alpha=None, minimum_weight=self.minimum_weight, minimum_delta=self.minimum_delta)

        means=np.random.normal(0,5,size=(ncomp, dim))
        diag_stds=np.random.uniform(0.5,2,size=(ncomp, dim))
        return {"dim":dim,
                "ncomp":ncomp,
                "weight":weights,
                "mean":means,
                "diag_std":diag_stds,
                }

    def sample(self, dic, n_samples=1000)->np.array:
        time0("sample")
        dim=dic["dim"]
        ncomp=dic["ncomp"]
        weights=dic["weight"]
        means=dic["mean"]
        diag_stds=dic["diag_std"]

        diag_stds=np.abs(diag_stds) 
        diag_stds=np.clip(diag_stds, a_min=1e-3, a_max=None)

        weights=np.abs(weights)
        weights=np.clip(weights, a_min=0.01, a_max=None)
        weights[np.isnan(weights)]=0.01
        weights[np.isinf(weights)]=0.01
        weights=weights/np.sum(weights)

        samples=np.zeros((n_samples, dim), dtype=np.float32)
        component_choices=np.random.choice(ncomp, size=n_samples, p=weights)
        for c in range(ncomp):
            indices=np.where(component_choices==c)[0]
            samples[indices]=np.random.normal(loc=means[c], scale=diag_stds[c], size=(len(indices), dim))
        if "rotation" in dic:
            #print("ROTATING")
            rotation=dic["rotation"]
            samples=samples@rotation.T
        return samples




    def sample_exact(self, dic, n_samples=1000)->np.array:
        dim=dic["dim"]
        ncomp=dic["ncomp"]
        weights=dic["weight"]
        means=dic["mean"]
        diag_stds=dic["diag_std"]

        diag_stds=np.abs(diag_stds) 
        diag_stds=np.clip(diag_stds, a_min=1e-3, a_max=None)

        weights=np.abs(weights)
        weights=np.clip(weights, a_min=0.01, a_max=None)
        weights[np.isnan(weights)]=0.01
        weights[np.isinf(weights)]=0.01
        weights=weights/np.sum(weights)
        
        samples=np.zeros((n_samples, dim), dtype=np.float32)

        component_counts=np.round(weights*n_samples).astype(int)
        while sum(component_counts) < n_samples:
            component_counts[np.argmax(weights)] += 1
        while sum(component_counts) > n_samples:
            component_counts[np.argmax(weights)] -= 1
        component_counts=np.clip(component_counts, a_min=0, a_max=None)
        component_choices=[]
        for c in range(ncomp):
            component_choices.extend([c]*component_counts[c])
        component_choices=np.array(component_choices)
        component_choices=np.random.permutation(component_choices)

        for c in range(ncomp):
            indices=np.where(component_choices==c)[0]
            samples[indices]=_sample_exact_gauss(means[c], diag_stds[c], len(indices))
        return samples

    def sample_torch(self, dic, n_samples=1000, temperature=0.2):
        dim = dic["dim"]
        K = dic["ncomp"]

        weights = dic["weight"]
        means = dic["mean"]
        diag_stds = dic["diag_std"]

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        if not isinstance(means, torch.Tensor):
            means = torch.tensor(means, dtype=torch.float32)
        if not isinstance(diag_stds, torch.Tensor):
            diag_stds = torch.tensor(diag_stds, dtype=torch.float32)

        device = means.device

        # positive stds
        diag_stds = torch.clamp(diag_stds.abs(), min=1e-3)

        # normalized weights
        weights = torch.clamp(weights.abs(), min=1e-2)
        weights[torch.isnan(weights)] = 1e-2
        weights = weights / weights.sum()

        # ---------- Gumbel-Softmax ----------
        logits = torch.log(weights)

        gumbel = -torch.log(-torch.log(torch.rand(n_samples, K, device=device)+1e-8)+1e-8)
        alpha = torch.softmax((logits + gumbel) / temperature, dim=1)
        # alpha.shape = (n_samples, K)

        # ---------- Reparameterized Gaussian ----------
        eps = torch.randn(n_samples, K, dim, device=device)

        component_samples = (
            means.unsqueeze(0)
            + diag_stds.unsqueeze(0) * eps
        )
        # (n_samples, K, dim)

        # ---------- Soft mixture ----------
        samples = (alpha.unsqueeze(-1) * component_samples).sum(dim=1)

        return samples


    def template_identifier(self, dic)->dict:
        return {"dim":dic["dim"],
                "ncomp":dic["ncomp"],
                }

    def meta_info(self, dic)->dict:
        return {"ncomp":dic["ncomp"],
                "dim":dic["dim"],
                "min_weight":np.min(dic["weight"]),
                "max_weight":np.max(dic["weight"]),
               }

    def _get_whiten_rotation(self,mean, diag_std, weights):
        """
        Compute a single global rotation matrix for a diagonal-covariance GMM.

        Parameters
        ----------
        mean : np.ndarray
            Shape (K, D)
        diag_std : np.ndarray
            Shape (K, D)
        weights : np.ndarray
            Shape (K,)

        Returns
        -------
        rotation : np.ndarray
            Shape (D, D), orthogonal rotation matrix.
        """
        mean = np.asarray(mean, dtype=np.float64)
        diag_std = np.asarray(diag_std, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        # Normalize weights
        weights = np.abs(weights)
        weights = np.clip(weights, 0.01, None)
        weights[np.isnan(weights)] = 0.01
        weights[np.isinf(weights)] = 0.01
        weights /= weights.sum()

        # Global mixture mean
        aggr_mean = np.sum(weights[:, None] * mean, axis=0)

        # Global covariance
        K, D = mean.shape
        covariance = np.zeros((D, D), dtype=np.float64)

        for k in range(K):
            delta = mean[k] - aggr_mean

            # Within-component covariance
            covariance += weights[k] * np.diag(diag_std[k] ** 2)

            # Between-component covariance
            covariance += weights[k] * np.outer(delta, delta)

        # Eigenvectors = global PCA rotation
        _, rotation = np.linalg.eigh(covariance)

        return rotation


    def normalize(self, dic, method="zscore"):
        if method=="zscore" or method=="whiten":
            mean=dic["mean"]
            std=dic["diag_std"]
            weights=dic["weight"]
            ret={key:val for key,val in dic.items() if key not in ["mean", "diag_std"]}
            if method=="whiten":
                rotation=self._get_whiten_rotation(mean, std, weights)
                ret["rotation"]=rotation
                mean=mean@rotation.T

            weights=np.abs(weights)
            weights=np.clip(weights, a_min=0.01, a_max=None)
            weights[np.isnan(weights)]=0.01
            weights=weights/np.sum(weights)

            aggr_mean=np.sum(weights[:, np.newaxis] * mean, axis=0)
            aggr_var=np.sum(weights[:, np.newaxis] * (std**2 + (mean - aggr_mean)**2), axis=0)
            aggr_std=np.sqrt(aggr_var)
            aggr_std=np.clip(aggr_std, a_min=1e-4, a_max=None)
            #subtract mean, then divide by std
            mean=(mean-aggr_mean)/aggr_std
            std=std/aggr_std
            
            ret["mean"]=mean
            ret["diag_std"]=std
            norm_info={"mean":aggr_mean, "std":aggr_std}
            if method=="whiten":
                inv_rotation=rotation.T
                norm_info={"mean":aggr_mean, "std":aggr_std, "rotation":rotation, "inv_rotation":inv_rotation}
            else:
                norm_info={"mean":aggr_mean, "std":aggr_std}
        else:
            raise NotImplementedError
        return ret, norm_info

    def denormalize(self, dic, norm_info, method="zscore")->dict:
        if method=="zscore" or method=="whiten":
            aggr_mean=norm_info["mean"]
            aggr_std=norm_info["std"]
            mean=dic["mean"]
            std=dic["diag_std"]
            #multiply by std, then add mean
            mean=mean*aggr_std+aggr_mean
            std=std*aggr_std
            ret={key:val for key,val in dic.items() if key not in ["mean", "diag_std"]}
            ret["mean"]=mean
            ret["diag_std"]=std
            if method=="whiten":
                rotation=ret["rotation"] if "rotation" in ret else np.eye(ret["dim"])
                ret["mean"]=ret["mean"] @ rotation
                rotation=rotation @ norm_info["inv_rotation"]
                ret["rotation"]=rotation
        else:
            raise NotImplementedError
        return ret

    def density(self, dic, samples, eps=1e-6)->np.array:
        N, dim = samples.shape
        K = dic["ncomp"]
        means = np.asarray(dic["mean"], dtype=np.float64)        # Shape: (K, dim)
        diag_stds = np.asarray(dic["diag_std"], dtype=np.float64)  # Shape: (K, dim)

        # 1. Standard deviation preprocessing
        diag_stds = np.abs(diag_stds)
        diag_stds = np.clip(diag_stds, a_min=1e-3, a_max=None)

        # 3. Weight normalization
        weights = np.abs(dic["weight"])
        weights = weights / np.sum(weights)  # Shape: (K,)
        log_weights = np.log(weights + 1e-12)

        # 4. Compute log likelihood
        X_expanded = samples[:, np.newaxis, :]           # Shape: (N, 1, dim)
        means_expanded = means[np.newaxis, :, :]   # Shape: (1, K, dim)
        stds_expanded = diag_stds[np.newaxis, :, :]# Shape: (1, K, dim)

        sq_dist = ((X_expanded - means_expanded) / stds_expanded) ** 2

        log_component_pdf = -0.5 * np.sum(
            np.log(2 * np.pi) + 2 * np.log(stds_expanded) + sq_dist, 
            axis=-1
        )

        # 5. Combine mixture weights and log component densities
        log_component_joint = log_weights[np.newaxis, :] + log_component_pdf
        log_p = logsumexp(log_component_joint, axis=-1)

        return log_p


    def score(self, dic, samples, eps=1e-6)->np.array:
        N, dim = samples.shape
        K = dic["ncomp"]
        means = np.asarray(dic["mean"], dtype=np.float64)
        diag_stds = np.asarray(dic["diag_std"], dtype=np.float64)

        diag_stds = np.abs(diag_stds)
        diag_stds = np.clip(diag_stds, a_min=1e-3, a_max=None)

        weights = np.abs(dic["weight"])
        weights = weights / np.sum(weights)
        log_weights = np.log(weights + 1e-12)

        X_exp = samples[:, np.newaxis, :]
        means_exp = means[np.newaxis, :, :]
        stds_exp = diag_stds[np.newaxis, :, :]

        sq_dist = ((X_exp - means_exp) / stds_exp) ** 2

        log_component_pdf = -0.5 * np.sum(
            np.log(2 * np.pi) + 2 * np.log(stds_exp) + sq_dist, 
            axis=-1
        )

        log_component_joint = log_weights[np.newaxis, :] + log_component_pdf
        log_p = logsumexp(log_component_joint, axis=-1, keepdims=True)
        
        gamma = np.exp(log_component_joint - log_p)
        component_scores = -(X_exp - means_exp) / (stds_exp ** 2)
        score = np.sum(gamma[:, :, np.newaxis] * component_scores, axis=1)

        return score

    def prep_optimization(self, dic):
        dic_torch={key:val for key,val in dic.items()}
        dic_torch["mean"]=torch.tensor(dic["mean"].copy(), dtype=torch.float32, requires_grad=True)
        dic_torch["diag_std"]=torch.tensor(dic["diag_std"].copy(), dtype=torch.float32, requires_grad=True)
        dic_torch["weight"]=torch.tensor(dic["weight"].copy(), dtype=torch.float32, requires_grad=True)
        variables=[dic_torch["mean"], dic_torch["diag_std"], dic_torch["weight"]]
        return dic_torch, variables


if __name__=="__main__":
    prior=GMMPrior()
    dist=prior.random_distribution(5,2)
    print(dist)
    for token in prior.describe(dist):
        print(token)




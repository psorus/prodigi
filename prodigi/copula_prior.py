from .prior import Prior
import numpy as np
import scipy
from scipy.stats import norm, t, beta, expon, uniform
import scipy.stats as stats
import scipy.special as special
from scipy.special import stdtrit
#from scipy.stats.sampling import NumericalInverseHermite
import torch
#from helper import time0, time1
from .fast_linear_approx import FastLinearApprox, FastLinearApprox2d
from functools import lru_cache
import math
from .lkj import random_lkj_matrix

from .exact_sample import _sample_exact_gauss

#norm_cdf_approx = FastLinearApprox(norm.cdf, -5, 5, 10000)
#stdtrit_approx = FastLinearApprox2d(stdtrit, 0.5, 12, 0.0001, 0.9999, 1000)

EPS = 1e-6

def kumaraswamy_ppf(u, a, b, eps=EPS):
    a = np.abs(a) + eps
    b = np.abs(b) + eps
    u = np.clip(u, eps, 1 - eps)
    return (1 - (1 - u) ** (1 / b)) ** (1 / a)


def kumaraswamy_ppf_torch(u, a, b, eps=EPS):
    a = torch.abs(a) + eps
    b = torch.abs(b) + eps
    u = torch.clamp(u, eps, 1 - eps)
    return (1 - (1 - u) ** (1 / b)) ** (1 / a)


def _kumaraswamy_moments(a, b, eps=EPS):
    a = np.abs(a) + eps
    b = np.abs(b) + eps
    m1 = b * special.beta(1.0 + 1.0 / a, b)
    m2 = b * special.beta(1.0 + 2.0 / a, b)
    s1 = np.sqrt(np.maximum(m2 - m1 ** 2, 1e-12))
    return m1, s1


def _kumaraswamy_moments_torch(a, b, eps=EPS):
    a = torch.abs(a) + eps
    b = torch.abs(b) + eps

    def beta_fn(x, y):
        return torch.exp(torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y))

    m1 = b * beta_fn(1.0 + 1.0 / a, b)
    m2 = b * beta_fn(1.0 + 2.0 / a, b)
    s1 = torch.sqrt(torch.clamp(m2 - m1 ** 2, min=1e-12))
    return m1, s1


def random_positive_definite_correlation_matrix(dim, n_samples=None):
    """Generates a random positive definite correlation matrix by sampling random data."""
    if n_samples is None:
        n_samples = dim + 1
    if n_samples <= 2:
        n_samples = 3
    data = np.random.randn(n_samples, dim)
    corr = np.corrcoef(data, rowvar=False)
    return corr

def _ppf_normal(u, mean, std, p2, p3, eps=EPS):
    std = np.maximum(np.abs(std),eps)
    return norm.ppf(u, loc=mean, scale=std)


def _ppf_uniform(u, mean, std, p2, p3, eps=EPS):
    std = np.maximum(np.abs(std),eps)
    half = std * np.sqrt(3.0)          # Uniform(-sqrt3,sqrt3) has mean 0, std 1
    low = mean - half
    return uniform.ppf(u, loc=low, scale=2 * half)


def _ppf_exponential(u, mean, std, p2, p3, eps=EPS):
    std = np.maximum(np.abs(std),eps)            # for Exponential: std == scale, always
    loc = mean - std                   # shifting decouples mean from std
    return expon.ppf(u, loc=loc, scale=std)


def _ppf_studentt(u, mean, std, df, p3, eps=EPS):
    std = np.maximum(np.abs(std),eps)
    df_eff = np.clip(np.abs(df), 2.0 + 1e-2, 12.0)   # keep finite variance + in table domain
    scale = std * np.sqrt((df_eff - 2.0) / df_eff)   # so Var(scale*T_df) == std**2
    return stdtrit(df_eff, u) * scale + mean


def _ppf_kumaraswamy(u, mean, std, a, b, eps=EPS):
    std = np.maximum(np.abs(std),eps)
    m1, s1 = _kumaraswamy_moments(a, b, eps)
    k = kumaraswamy_ppf(u, a, b, eps)
    return mean + std * (k - m1) / s1


ppf_marginals = {
    "normal": _ppf_normal,
    "uniform": _ppf_uniform,
    "exponential": _ppf_exponential,
    "studentt": _ppf_studentt,
    "kumaraswamy": _ppf_kumaraswamy,   
}

allowed_functions = list(ppf_marginals.keys())


def _ppf_normal_torch(u, mean, std, p2, p3, eps=EPS):
    std = torch.abs(std)
    if std<eps:
        std = eps
    return mean + std * math.sqrt(2.0) * torch.erfinv(2 * u - 1)


def _ppf_uniform_torch(u, mean, std, p2, p3, eps=EPS):
    std = torch.abs(std)
    if std<eps:
        std = eps
    half = std * math.sqrt(3.0)
    return mean + half * (2 * u - 1)


def _ppf_exponential_torch(u, mean, std, p2, p3, eps=EPS):
    std = torch.abs(std)
    if std<eps:
        std = eps
    loc = mean - std
    return loc - std * torch.log1p(-u)


def _ppf_studentt_torch(u, mean, std, df, p3, eps=EPS):
    std = torch.abs(std)
    if std<eps:
        std = eps
    df_eff = torch.clamp(torch.abs(df), min=2.0 + 1e-2, max=12.0)
    scale = std * torch.sqrt((df_eff - 2.0) / df_eff)
    base = torch.tensor(
        stdtrit(df_eff.detach().numpy(), u.detach().numpy()),
        dtype=torch.float32,
    )
    return base * scale + mean


def _ppf_kumaraswamy_torch(u, mean, std, a, b, eps=EPS):
    std = torch.abs(std)
    if std<eps:
        std = eps
    m1, s1 = _kumaraswamy_moments_torch(a, b, eps)
    k = kumaraswamy_ppf_torch(u, a, b, eps)
    return mean + std * (k - m1) / s1


torch_ppf_marginals = {
    "normal": _ppf_normal_torch,
    "uniform": _ppf_uniform_torch,
    "exponential": _ppf_exponential_torch,
    "studentt": _ppf_studentt_torch,
    "kumaraswamy": _ppf_kumaraswamy_torch,
}


def student_t_cdf_torch(x, df, n_points=64):
    """
    Differentiable approximation to Student-t CDF.

    x  : arbitrary shape
    df : scalar or broadcastable to x

    Uses Gauss-Legendre quadrature on [0, |x|].
    """

    dtype = x.dtype
    device = x.device

    # Standard t PDF.
    def pdf(z):
        log_norm = (
            torch.lgamma((df + 1.0) / 2.0)
            - torch.lgamma(df / 2.0)
            - 0.5 * torch.log(df * torch.pi)
        )

        log_pdf = (
            log_norm
            - 0.5 * (df + 1.0)
            * torch.log1p(z ** 2 / df)
        )

        return torch.exp(log_pdf)

    ax = torch.abs(x)

    # Gauss-Legendre nodes/weights.
    nodes, weights = np.polynomial.legendre.leggauss(n_points)

    nodes = torch.tensor(
        nodes,
        dtype=dtype,
        device=device,
    )

    weights = torch.tensor(
        weights,
        dtype=dtype,
        device=device,
    )

    # Map [-1,1] -> [0, |x|].
    z = (
        ax.unsqueeze(-1)
        * (nodes + 1.0)
        / 2.0
    )

    integral = (
        ax.unsqueeze(-1)
        / 2.0
        * weights
        * pdf(z)
    ).sum(dim=-1)

    # T(0) = 0.5 and symmetry.
    cdf = 0.5 + torch.sign(x) * integral

    return cdf


def correlation_from_values(values, dim, eps=1e-4):
    R = np.eye(dim)
    iu = triu_indices(dim)
    R[iu] = values
    R[(iu[1], iu[0])] = values

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, eps)
    R = eigvecs @ np.diag(eigvals) @ eigvecs.T

    d = np.sqrt(np.diag(R))
    R /= np.outer(d, d)
    return R

# def correlation_from_values_torch(values, dim, eps=1e-4):
#     R = torch.eye(dim, device=values.device)
#     iu = triu_indices(dim)
#     R[iu] = values
#     R[(iu[1], iu[0])] = values
#
#     eigvals, eigvecs = torch.linalg.eigh(R)
#     eigvals = torch.clamp(eigvals, min=eps)
#     R = eigvecs @ torch.diag(eigvals) @ eigvecs.T
#
#     d = torch.sqrt(torch.diagonal(R))
#     R /= d.unsqueeze(1) * d.unsqueeze(0)
#     return R
def correlation_from_values_torch(values, dim, eps=1e-4):
    """
    Differentiable construction of a correlation matrix from its
    upper-triangular off-diagonal entries.

    values: shape (D*(D-1)/2,)
    """

    device = values.device
    dtype = values.dtype

    # Indices of upper triangle, excluding diagonal.
    iu = torch.triu_indices(
        dim,
        dim,
        offset=1,
        device=device,
    )

    R = torch.eye(dim, device=device, dtype=dtype)

    # Construct symmetric matrix without in-place writes.
    upper = torch.zeros(
        (dim, dim),
        device=device,
        dtype=dtype,
    )

    upper = upper.index_put(
        (iu[0], iu[1]),
        values,
    )

    R = R + upper + upper.T

    # Project to PSD by clipping eigenvalues.
    eigvals, eigvecs = torch.linalg.eigh(R)
    eigvals = torch.clamp(eigvals, min=eps)

    R = eigvecs @ torch.diag(eigvals) @ eigvecs.T

    # Renormalize diagonal to one.
    d = torch.sqrt(torch.clamp(torch.diagonal(R), min=eps))
    R = R / (d[:, None] * d[None, :])

    return R

def correlation_from_values_torch(values, dim, eps=1e-4):
    """
    Differentiable equivalent of correlation_from_values().
    """

    device = values.device
    dtype = values.dtype

    iu = torch.triu_indices(
        dim,
        dim,
        offset=1,
        device=device,
    )

    if values.ndim == 2:
        values = values[iu[0], iu[1]]

    elif values.ndim != 1:
        raise ValueError(
            f"Expected 1D or 2D values, got {values.shape}"
        )

    expected = dim * (dim - 1) // 2

    if values.numel() != expected:
        raise ValueError(
            f"Expected {expected} values, got {values.numel()}"
        )

    if not torch.isfinite(values).all():
        #make nonfinite values=0
        #does not work!
        #values[~torch.isfinite(values)] = 0.0
        raise FloatingPointError(
            "Correlation parameters contain NaN/Inf"
        )

    # ---------------------------------------------------------------
    # Same raw symmetric matrix as NumPy.
    # ---------------------------------------------------------------
    upper = torch.zeros(
        (dim, dim),
        device=device,
        dtype=dtype,
    )

    upper = upper.index_put(
        (iu[0], iu[1]),
        values,
    )

    R = (
        torch.eye(
            dim,
            device=device,
            dtype=dtype,
        )
        + upper
        + upper.T
    )

    R = 0.5 * (R + R.T)

    # ---------------------------------------------------------------
    # Same eigenvalue projection as NumPy.
    # ---------------------------------------------------------------
    eigvals, eigvecs = torch.linalg.eigh(R)

    eigvals = torch.clamp(
        eigvals,
        min=eps,
    )

    # More numerically stable than constructing diag(eigvals).
    R = (
        eigvecs * eigvals.unsqueeze(0)
    ) @ eigvecs.T

    # ---------------------------------------------------------------
    # Same diagonal normalization as NumPy.
    # ---------------------------------------------------------------
    d = torch.sqrt(
        torch.clamp(
            torch.diagonal(R),
            min=eps,
        )
    )

    R = R / (
        d[:, None] * d[None, :]
    )

    return 0.5 * (R + R.T)



@lru_cache(maxsize=None)
def triu_indices(dim):
    return np.triu_indices(dim, k=1)


def upper_triangle(matrix):
    triu_ind = triu_indices(matrix.shape[1])
    return matrix[triu_ind]

def upper_triangle_torch(matrix):
    triu_ind = triu_indices(matrix.shape[1])
    return matrix[triu_ind]


def from_upper_triangle(triag, dim):
    triag=np.asarray(triag)
    if len(triag.shape) > 1:
        triag = upper_triangle(triag)
    return correlation_from_values(triag, dim)

def from_upper_triangle_torch(triag, dim):
    if len(triag.shape) > 1:
        triag = upper_triangle_torch(triag)
    return correlation_from_values_torch(triag, dim)

class CopulaPrior(Prior):
    def __init__(self, cfg=None):
        super().__init__(cfg, "copula_pattern.boso")
        self.advanced_corr=self.cfg.ADVANCED_CORR
        self.eta_min=self.cfg.ETA_MIN
        self.eta_max=self.cfg.ETA_MAX

    def allowed_args(self, cfg=None) -> list:
        if cfg is None:cfg=self.cfg
        MIN_DIM, MAX_DIM = cfg.MIN_DIM, cfg.MAX_DIM
        lis = []
        for dim in range(MIN_DIM, MAX_DIM + 1):
            lis.append({"dim": dim})
        return lis

    def random_distribution(self, dim=2, **kwargs):
        if self.advanced_corr:
            corr= random_lkj_matrix(dim, self.eta_min, self.eta_max)
        else:
            corr = random_positive_definite_correlation_matrix(dim)
        corrcount = (dim * (dim - 1)) // 2

        function = np.random.choice(allowed_functions, size=dim, replace=True)

        params = []
        for f in function:
            mean = np.random.uniform(-5, 5)
            std = np.random.uniform(0.5, 2)
            if not f.lower().strip() in allowed_functions:
                f="normal"

            if f == "normal":
                shape = [0.0, 0.0]
            elif f == "uniform":
                shape = [0.0, 0.0]
            elif f == "exponential":
                shape = [0.0, 0.0]
            elif f == "studentt":
                df = np.random.uniform(2.5, 10)
                shape = [df, 0.0]
            elif f == "kumaraswamy":
                a = np.random.uniform(0.5, 5)
                b = np.random.uniform(0.5, 5)
                shape = [a, b]
            else:
                raise ValueError(f"Unknown marginal function: {f}")

            params.append([mean, std] + shape)
        params = np.array(params)

        return {
            "dim": dim,
            "corrcount": corrcount,
            "function": function,
            "params": params,   
            "corr": corr,
        }

    def posthoc(self, dic):
        return True
        function = dic["function"]
        for f in function:
            if not str(f).lower() in ppf_marginals:
                return False
        return True

    def posthoc_debug(self, dic):
        dim, corr = dic["dim"], dic["corr"]
        matrix = from_upper_triangle(corr, dim)
        np.linalg.cholesky(matrix)
        function = dic["function"]
        for f in function:
            if not str(f).lower() in ppf_marginals:
                f="normal"
                #raise ValueError(f"Unknown marginal function: {f}")
        assert np.all(np.linalg.eigvals(matrix) > 0), "Matrix is not positive definite"

    def _sample_common(self, dic, n_samples, z_generator):
        dim, function, params = dic["dim"], dic["function"], dic["params"]
        params=np.array(params)
        corr = dic["corr"] if dim > 1 else np.eye(dim)
        corr = from_upper_triangle(corr, dim)

        z = z_generator(dim, n_samples)
        y = z @ np.linalg.cholesky(corr).T
        u = norm.cdf(y)

        samples = np.zeros_like(u)
        functions = np.array([str(f).lower() for f in function])
        functions = np.array([zw if zw in ppf_marginals else "normal" for zw in functions])
        for func in set(functions):
            indices = np.where(functions == func)[0]
            p = params[indices]  # shape (k, 4)
            samples[:, indices] = ppf_marginals[func](
                u[:, indices], p[:, 0], p[:, 1], p[:, 2], p[:, 3]
            )
        return samples

    def sample(self, dic, n_samples=1000) -> np.array:
        return self._sample_common(
            dic, n_samples,
            z_generator=lambda dim, n: np.random.normal(size=(n, dim)),
        )

    def sample_exact(self, dic, n_samples=1000) -> np.array:
        return self._sample_common(
            dic, n_samples,
            z_generator=lambda dim, n: _sample_exact_gauss(np.zeros(dim), np.ones(dim), n),
        )

    def sample_torch(self, dic, n_samples=1000):
        dim = dic["dim"]
        function = dic["function"]
        params = dic["params"]
        corr = dic["corr"] if dim > 1 else torch.eye(dim)

        if not isinstance(corr, torch.Tensor):
            corr = torch.tensor(corr, dtype=torch.float32)
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=torch.float32)

        device = corr.device

        corr = dic["corr"] if dim > 1 else torch.eye(dim)
        corr = from_upper_triangle_torch(corr, dim)

        z = torch.randn(n_samples, dim, device=device)
        y = z @ torch.linalg.cholesky(corr).T

        u = 0.5 * (1.0 + torch.erf(y / np.sqrt(2.0)))
        u = torch.clamp(u, 1e-7, 1 - 1e-7)

        samples = torch.empty_like(u)
        for i in range(dim):
            f = str(function[i]).lower()
            if not f in torch_ppf_marginals:
                f="normal"
                #raise ValueError(f"Unsupported marginal function: {f}")
            samples[:, i] = torch_ppf_marginals[f](u[:, i], *params[i])

        return samples

    def template_identifier(self, dic) -> dict:
        return {"dim": dic["dim"]}

    def meta_info(self, dic) -> dict:
        return {"dim": dic["dim"]}

    def normalize(self, dic, method="zscore"):
        if method != "zscore":
            raise NotImplementedError

        params = dic["params"]
        mean = params[:, 0].copy()
        std = np.clip(np.abs(params[:, 1]), 1e-4, None)

        new_params = params.copy()
        new_params[:, 0] = 0.0
        new_params[:, 1] = 1.0

        ret = {key: val for key, val in dic.items()}
        ret["params"] = new_params
        norm_info = {"mean": mean, "std": std}
        return ret, norm_info

    def denormalize(self, dic, norm_info, method="zscore") -> dict:
        if method != "zscore":
            raise NotImplementedError

        params = dic["params"].copy()
        params[:, 0] = norm_info["mean"]
        params[:, 1] = norm_info["std"]

        ret = {key: val for key, val in dic.items()}
        ret["params"] = params
        return ret

    def _parse_functions_and_params(self, dic, dim):
        """Helper to parse and broadcast functions and params to match dimension size."""
        raw_funcs = dic["function"]
        if isinstance(raw_funcs, (str, bytes)):
            functions = [str(raw_funcs).lower()] * dim
        elif len(raw_funcs) == 1:
            functions = [str(raw_funcs[0]).lower()] * dim
        else:
            functions = [str(f).lower() for f in raw_funcs]

        raw_params = dic["params"]
        if len(raw_params) == 1 and dim > 1:
            params = [raw_params[0]] * dim
        else:
            params = raw_params

        return functions, params


    def density(self, dic, samples, eps=1e-6)->np.array:
        N, dim = samples.shape
        functions, params = self._parse_functions_and_params(dic, dim)
        R = dic["corr"] if dim > 1 else np.eye(dim)
        R = from_upper_triangle(R, dim)
        R_inv = np.linalg.inv(R)

        u = np.zeros((N, dim), dtype=np.float64)
        log_pdf_x = np.zeros((N, dim), dtype=np.float64)

        for i in range(dim):
            f = functions[i]
            if not f in ppf_marginals:
                f="normal"
            p = params[i]
            x_col = samples[:, i]

            if f == "normal":
                mu, sigma = p[0], abs(p[1]) + eps
                u[:, i] = norm.cdf(x_col, loc=mu, scale=sigma)
                log_pdf_x[:, i] = norm.logpdf(x_col, loc=mu, scale=sigma)

            elif f == "studentt":
                #df, loc, scale = p[0], p[1], abs(p[2]) + eps
                #u[:, i] = t.cdf(x_col, df=df, loc=loc, scale=scale)
                #log_pdf_x[:, i] = t.logpdf(x_col, df=df, loc=loc, scale=scale)
                mu, std = p[0], abs(p[1])
                df = np.clip(abs(p[2]), 2.0 + 1e-2, 12.0)

                scale = std * np.sqrt((df - 2.0) / df)

                u[:, i] = t.cdf(x_col, df=df, loc=mu, scale=scale)
                log_pdf_x[:, i] = t.logpdf(x_col, df=df, loc=mu, scale=scale)

            elif False and f == "kumaraswamy":
                mu, sigma = p[0], abs(p[1])
                a, b = abs(p[0]) + eps, abs(p[1]) + eps
                #x_clipped = np.clip(x_col, eps, 1.0 - eps)
                #u[:, i] = 1.0 - (1.0 - x_clipped ** a) ** b
                #pdf_val = a * b * (x_clipped ** (a - 1.0)) * ((1.0 - x_clipped ** a) ** (b - 1.0))
                #log_pdf_x[:, i] = np.log(np.maximum(pdf_val, 1e-12))

                m1, s1 = _kumaraswamy_moments(a, b)

                k = m1 + s1 * (x_col - mu) / sigma

                log_pdf_k = (
                    np.log(a)
                    + np.log(b)
                    + (a - 1.0) * np.log(k)
                    + (b - 1.0) * np.log1p(-k**a)
                )

                log_pdf_x = log_pdf_k - np.log(sigma * s1)
            elif False and f == "kumaraswamy":
                mu = p[0]
                sigma = max(abs(p[1]), eps)
                a = abs(p[2]) + eps
                b = abs(p[3]) + eps

                m1, s1 = _kumaraswamy_moments(a, b)

                # Transform x back to the underlying Kumaraswamy variable.
                k = m1 + s1 * (x_col - mu) / sigma

                # Support check.
                valid = (k > 0.0) & (k < 1.0)

                u[:, i] = np.nan
                log_pdf_x[:, i] = -np.inf

                kv = k[valid]

                # CDF
                u[valid, i] = 1.0 - (1.0 - kv**a)**b

                # log PDF, including affine Jacobian
                log_pdf_x[valid, i] = (
                    np.log(a)
                    + np.log(b)
                    + (a - 1.0) * np.log(kv)
                    + (b - 1.0) * np.log1p(-kv**a)
                    - np.log(sigma * s1)
                )

            elif f == "kumaraswamy":
                mu, sigma, a, b = p

                sigma = max(abs(sigma), eps)
                a = max(abs(a), eps)
                b = max(abs(b), eps)

                m1, s1 = _kumaraswamy_moments(a, b)

                # Standardized Kumaraswamy coordinate
                k = m1 + (x_col - mu) * s1 / sigma

                valid = (k > 0.0) & (k < 1.0)

                u[:, i] = np.nan
                log_pdf_x[:, i] = -np.inf

                kv = k[valid]

                # CDF
                u[valid, i] = 1.0 - (1.0 - kv**a)**b

                # log f_K(k)
                log_pdf_k = (
                    np.log(a)
                    + np.log(b)
                    + (a - 1.0) * np.log(kv)
                    + (b - 1.0) * np.log1p(-(kv**a))
                )

                # dk/dx = s1 / sigma
                log_pdf_x[valid, i] = log_pdf_k + np.log(s1) - np.log(sigma)



            elif f == "exponential":
                #rate = abs(p[0]) + eps
                #scale = 1.0 / rate
                #u[:, i] = expon.cdf(x_col, scale=scale)
                #log_pdf_x[:, i] = expon.logpdf(x_col, scale=scale)
                mu, sigma = p[0], abs(p[1])
                loc = mu - sigma

                u[:, i] = expon.cdf(x_col, loc=loc, scale=sigma)
                log_pdf_x[:, i] = expon.logpdf(x_col, loc=loc, scale=sigma)

            elif f == "uniform":
                mu, sigma = p[0], abs(p[1])
                #low, high = min(p[0], p[1]), max(p[0], p[1])
                #u[:, i] = uniform.cdf(x_col, loc=low, scale=abs(high - low))
                #log_pdf_x[:, i] = uniform.logpdf(x_col, loc=low, scale=abs(high - low)))
                half = sigma * np.sqrt(3.0)
                low = mu - half
                high = mu + half

                u[:, i] = uniform.cdf(x_col, loc=low, scale=high-low)
                log_pdf_x[:, i] = uniform.logpdf(x_col, loc=low, scale=high-low)

            else:
                raise ValueError(f"Unsupported marginal function: {f}")

        u = np.clip(u, eps, 1.0 - eps)

        y = norm.ppf(u)

        sign, logdet = np.linalg.slogdet(R)
        y_Rinv = y @ R_inv.T
        quad_term = np.sum(y * y_Rinv, axis=1) - np.sum(y ** 2, axis=1)
        log_copula_density = -0.5 * logdet - 0.5 * quad_term

        log_p = log_copula_density + np.sum(log_pdf_x, axis=1)
        return log_p

    def density_torch(self, dic, samples, eps=1e-6):
        """
        Differentiable log-density of the Gaussian copula distribution.

        Gradients are preserved with respect to:

            dic["params"]  -- shape (D, 4)
            dic["corr"]    -- shape (D*(D-1)/2,)

        Parameterization:

            normal:
                params[i] = [mu, std, _, _]

            uniform:
                params[i] = [mu, std, _, _]
                X ~ Uniform(mu - sqrt(3)*std,
                            mu + sqrt(3)*std)

            exponential:
                params[i] = [mu, std, _, _]
                X = (mu - std) + std * Exp(1)

            studentt:
                params[i] = [mu, std, df, _]
                X = mu + scale * T_df

                scale = std * sqrt((df - 2) / df)

            kumaraswamy:
                params[i] = [mu, std, a, b]
                X = mu + std * (K - m1) / s1

        Returns:
            log_p : shape (N,)
        """

        params = dic["params"]

        if not isinstance(samples, torch.Tensor):
            samples = torch.as_tensor(
                samples,
                dtype=params.dtype,
                device=params.device,
            )

        samples = samples.to(
            dtype=params.dtype,
            device=params.device,
        )

        N, dim = samples.shape

        # ------------------------------------------------------------
        # Correlation matrix
        # ------------------------------------------------------------

        if dim > 1:
            corr_values = dic["corr"]

            if not isinstance(corr_values, torch.Tensor):
                corr_values = torch.as_tensor(
                    corr_values,
                    dtype=params.dtype,
                    device=params.device,
                )

            corr_values = corr_values.to(
                dtype=params.dtype,
                device=params.device,
            )

            R = from_upper_triangle_torch(
                corr_values,
                dim,
            )
        else:
            R = torch.ones(
                (1, 1),
                dtype=params.dtype,
                device=params.device,
            )

        # ------------------------------------------------------------
        # Marginal transforms
        # ------------------------------------------------------------

        u_list = []
        log_pdf_list = []

        sqrt2 = math.sqrt(2.0)
        sqrt3 = math.sqrt(3.0)

        for i in range(dim):

            f = str(self._parse_functions_and_params(dic, dim)[0][i]).lower()

            # Invalid functions have the same fallback as sample_torch.
            if f not in torch_ppf_marginals:
                f = "normal"

            p = params[i]
            x = samples[:, i]

            # --------------------------------------------------------
            # Normal
            # --------------------------------------------------------

            if f == "normal":

                mu = p[0]
                sigma = torch.clamp(
                    torch.abs(p[1]),
                    min=eps,
                )

                z = (x - mu) / sigma

                u = 0.5 * (
                    1.0 + torch.erf(z / sqrt2)
                )

                log_pdf = (
                    -0.5 * z ** 2
                    - torch.log(sigma)
                    - 0.5 * math.log(2.0 * math.pi)
                )

            # --------------------------------------------------------
            # Uniform
            # --------------------------------------------------------

            elif f == "uniform":

                mu = p[0]
                sigma = torch.clamp(
                    torch.abs(p[1]),
                    min=eps,
                )

                half = sigma * sqrt3

                low = mu - half
                high = mu + half

                u = (x - low) / (high - low)

                # The density is constant on the support.
                log_pdf = -torch.log(high - low)

                # Samples generated from this distribution should be
                # inside the support. For arbitrary NLL data, explicitly
                # represent outside-support points as -inf.
                inside = (
                    (x >= low)
                    & (x <= high)
                )

                #log_pdf = torch.where(
                #    inside,
                #    log_pdf,
                #    torch.full_like(log_pdf, -torch.inf),
                #)
                # Distance outside the support.
                violation = (
                    torch.relu(low - x)
                    + torch.relu(x - high)
                )

                # Smooth finite penalty.
                support_penalty = 50.0 * violation ** 2

                log_pdf = (
                    -torch.log(high - low)
                    - support_penalty
                )

            # --------------------------------------------------------
            # Exponential
            # --------------------------------------------------------

            elif f == "exponential":

                mu = p[0]
                sigma = torch.clamp(
                    torch.abs(p[1]),
                    min=eps,
                )

                loc = mu - sigma

                z = (x - loc) / sigma

                u = 1.0 - torch.exp(-z)

                log_pdf = (
                    -torch.log(sigma)
                    - z
                )

                inside = x >= loc

                #log_pdf = torch.where(
                #    inside,
                #    log_pdf,
                #    torch.full_like(log_pdf, -torch.inf),
                #)
                violation = torch.relu(loc - x)

                support_penalty = 50.0 * violation ** 2

                log_pdf = (
                    -torch.log(sigma)
                    - z
                    - support_penalty
                )

            # --------------------------------------------------------
            # Student-t
            # --------------------------------------------------------

            elif f == "studentt":

                mu = p[0]

                sigma = torch.clamp(
                    torch.abs(p[1]),
                    min=eps,
                )

                df = torch.clamp(
                    torch.abs(p[2]),
                    min=2.0 + 1e-2,
                    max=12.0,
                )

                # Convert desired standard deviation into t scale.
                scale = (
                    sigma
                    * torch.sqrt(
                        (df - 2.0) / df
                    )
                )

                z = (x - mu) / scale

                # Differentiable CDF.
                u = student_t_cdf_torch(
                    z,
                    df,
                )

                # Student-t log PDF.
                log_pdf = (
                    torch.lgamma((df + 1.0) / 2.0)
                    - torch.lgamma(df / 2.0)
                    - torch.log(scale)
                    - 0.5 * torch.log(df * math.pi)
                    - 0.5 * (df + 1.0)
                    * torch.log1p(z ** 2 / df)
                )

            # --------------------------------------------------------
            # Kumaraswamy
            # --------------------------------------------------------

            elif f == "kumaraswamy":

                mu = p[0]

                sigma = torch.clamp(
                    torch.abs(p[1]),
                    min=eps,
                )

                a = torch.abs(p[2]) + eps
                b = torch.abs(p[3]) + eps

                # ---------------------------------------------------------
                # Kumaraswamy(a, b) moments
                # ---------------------------------------------------------
                m1, s1 = _kumaraswamy_moments_torch(
                    a,
                    b,
                    eps,
                )

                # ---------------------------------------------------------
                # Location/scale transformation
                #
                # x = mu + sigma * (k - m1) / s1
                #
                # => k = m1 + (x - mu) * s1 / sigma
                # ---------------------------------------------------------
                k = (
                    m1
                    + (x - mu) * s1 / sigma
                )

                # ---------------------------------------------------------
                # Support violation
                #
                # Kumaraswamy has support:
                #
                #     0 < k < 1
                #
                # Do NOT return -inf during optimization. That causes the
                # NLL to become inf as soon as an optimizer step moves the
                # support away from a data point.
                # ---------------------------------------------------------
                violation = (
                    torch.relu(-k)
                    + torch.relu(k - 1.0)
                )

                inside = (
                    (k > 0.0)
                    & (k < 1.0)
                )

                # ---------------------------------------------------------
                # Numerically safe value for evaluating log terms.
                #
                # This is only used inside log/log1p. The actual k is still
                # used for the support penalty above.
                # ---------------------------------------------------------
                k_safe = torch.clamp(
                    k,
                    min=eps,
                    max=1.0 - eps,
                )

                k_a = k_safe ** a

                # ---------------------------------------------------------
                # Kumaraswamy CDF
                #
                #     F(k) = 1 - (1 - k^a)^b
                #
                # Use the clamped k so this remains finite even when the
                # optimizer temporarily moves k outside [0, 1].
                # ---------------------------------------------------------
                u = (
                    1.0
                    - (1.0 - k_a) ** b
                )

                # ---------------------------------------------------------
                # log f_K(k)
                #
                #     log(a)
                #   + log(b)
                #   + (a-1) log(k)
                #   + (b-1) log(1-k^a)
                # ---------------------------------------------------------
                log_pdf_k = (
                    torch.log(a)
                    + torch.log(b)
                    + (a - 1.0) * torch.log(k_safe)
                    + (b - 1.0)
                    * torch.log1p(-k_a)
                )

                # ---------------------------------------------------------
                # Jacobian:
                #
                #     dk/dx = s1 / sigma
                # ---------------------------------------------------------
                log_pdf = (
                    log_pdf_k
                    + torch.log(s1)
                    - torch.log(sigma)
                )

                # ---------------------------------------------------------
                # Differentiable support penalty
                #
                # Inside the support:
                #
                #     penalty = 0
                #
                # Outside:
                #
                #     penalty grows quadratically with distance outside
                #     the valid Kumaraswamy support.
                #
                # This is a surrogate likelihood used during optimization.
                # It prevents -inf from poisoning the NLL and gives the
                # optimizer a gradient that pushes the support back over x.
                # ---------------------------------------------------------
                support_penalty = (
                    violation ** 2
                )

                log_pdf = (
                    log_pdf
                    - support_penalty
                )

                # ---------------------------------------------------------
                # Keep the CDF finite and inside (0, 1).
                #
                # This is important because the next stage computes
                #     y = Phi^{-1}(u)
                # ---------------------------------------------------------
                u = torch.clamp(
                    u,
                    min=eps,
                    max=1.0 - eps,
                )

                # ---------------------------------------------------------
                # NOTE:
                #
                # If you want exact density semantics instead of an
                # optimization-friendly surrogate, replace the block above
                # with:
                #
                # log_pdf = torch.where(
                #     inside,
                #     log_pdf,
                #     torch.full_like(log_pdf, -torch.inf),
                # )
                #
                # But do NOT use that version while optimizing parameters
                # with NLL.
                # ---------------------------------------------------------

            else:
                raise ValueError(
                    f"Unsupported marginal function: {f}"
                )

            u_list.append(u)
            log_pdf_list.append(log_pdf)

        # ------------------------------------------------------------
        # Stack marginal quantities
        # ------------------------------------------------------------

        u = torch.stack(
            u_list,
            dim=1,
        )

        log_pdf_x = torch.stack(
            log_pdf_list,
            dim=1,
        )

        # ------------------------------------------------------------
        # Transform marginal CDFs -> Gaussian latent variables
        # ------------------------------------------------------------

        u = torch.clamp(
            u,
            min=eps,
            max=1.0 - eps,
        )

        y = sqrt2 * torch.erfinv(
            2.0 * u - 1.0
        )

        # ------------------------------------------------------------
        # Gaussian copula
        # ------------------------------------------------------------

        # R^{-1}
        #
        # Using solve rather than explicit inverse is numerically
        # preferable and remains differentiable.

        R_inv_y = torch.linalg.solve(
            R,
            y.T,
        ).T

        # y^T R^{-1} y
        quad = torch.sum(
            y * R_inv_y,
            dim=1,
        )

        # y^T y
        independent_quad = torch.sum(
            y ** 2,
            dim=1,
        )

        # Gaussian copula quadratic contribution:
        #
        # -1/2 y^T (R^{-1} - I) y

        quad_term = (
            quad
            - independent_quad
        )

        # log |R|
        sign, logdet = torch.linalg.slogdet(R)

        # Gaussian copula log density.
        log_copula = (
            -0.5 * logdet
            -0.5 * quad_term
        )

        # ------------------------------------------------------------
        # Final log density
        # ------------------------------------------------------------

        log_p = (
            log_copula
            + torch.sum(
                log_pdf_x,
                dim=1,
            )
        )

        return log_p







    def score(self, dic, samples, eps=1e-6)->np.array:
        """
        Computes the score function grad_x log p(x) for a Gaussian Copula.
        """
        N, dim = X.shape
        functions, params = self._parse_functions_and_params(dic, dim)
        corr = dic.get("corr", np.eye(dim))

        R = dic["corr"] if dim > 1 else np.eye(dim)
        R = from_upper_triangle(R, dim)
        R_inv = np.linalg.inv(R)

        u = np.zeros((N, dim), dtype=np.float64)
        pdf_x = np.zeros((N, dim), dtype=np.float64)
        score_marginal = np.zeros((N, dim), dtype=np.float64)

        for i in range(dim):
            f = functions[i]
            if not f in ppf_marginals:
                f="normal"
            p = params[i]
            x_col = X[:, i]

            if f == "normal":
                mu, sigma = p[0], abs(p[1]) + eps
                u[:, i] = norm.cdf(x_col, loc=mu, scale=sigma)
                pdf_x[:, i] = norm.pdf(x_col, loc=mu, scale=sigma)
                score_marginal[:, i] = -(x_col - mu) / (sigma ** 2)

            elif f == "studentt":
                df, loc, scale = p[0], p[1], abs(p[2]) + eps
                u[:, i] = t.cdf(x_col, df=df, loc=loc, scale=scale)
                pdf_x[:, i] = t.pdf(x_col, df=df, loc=loc, scale=scale)
                z = (x_col - loc) / scale
                score_marginal[:, i] = -((df + 1.0) * z) / ((df + z ** 2) * scale)

            elif f == "kumaraswamy":
                a, b = abs(p[0]) + eps, abs(p[1]) + eps
                x_clipped = np.clip(x_col, eps, 1.0 - eps)
                u[:, i] = 1.0 - (1.0 - x_clipped ** a) ** b
                pdf_x[:, i] = a * b * (x_clipped ** (a - 1.0)) * ((1.0 - x_clipped ** a) ** (b - 1.0))
                score_marginal[:, i] = (a - 1.0) / x_clipped - (b - 1.0) * a * (x_clipped ** (a - 1.0)) / (1.0 - x_clipped ** a)

            elif f == "exponential":
                rate = abs(p[0]) + eps
                scale = 1.0 / rate
                u[:, i] = expon.cdf(x_col, scale=scale)
                pdf_x[:, i] = expon.pdf(x_col, scale=scale)
                score_marginal[:, i] = -rate

            elif f == "uniform":
                low, high = min(p[0], p[1]), max(p[0], p[1])
                u[:, i] = uniform.cdf(x_col, loc=low, scale=abs(high - low))
                pdf_x[:, i] = uniform.pdf(x_col, loc=low, scale=abs(high - low))
                score_marginal[:, i] = 0.0

            else:
                raise ValueError(f"Unsupported marginal function: {f}")

        u = np.clip(u, eps, 1.0 - eps)
        pdf_x = np.maximum(pdf_x, 1e-12)

        y = norm.ppf(u)
        phi_y = norm.pdf(y)

        y_Rinv = y @ R_inv.T
        dlogc_du = -(y_Rinv - y) / np.maximum(phi_y, 1e-12)

        score = dlogc_du * pdf_x + score_marginal
        return score

    def score(self, dic, samples, eps=1e-6) -> np.array:
        """
        Computes the score function grad_x log p(x) for a Gaussian Copula.

        Parameterization:
            normal:
                p = [mu, sigma, _, _]

            uniform:
                p = [mu, sigma, _, _]
                X ~ Uniform(mu - sqrt(3)*sigma, mu + sqrt(3)*sigma)

            exponential:
                p = [mu, sigma, _, _]
                X = (mu - sigma) + sigma * Exp(1)

            studentt:
                p = [mu, sigma, df, _]
                X = mu + scale * T_df
                where scale = sigma * sqrt((df - 2) / df)

            kumaraswamy:
                p = [mu, sigma, a, b]
                X = mu + sigma * (K - m1) / s1
                where K ~ Kumaraswamy(a, b)

        Returns:
            score = grad_x log p(x), shape (N, dim)
        """
        N, dim = samples.shape

        functions, params = self._parse_functions_and_params(dic, dim)

        R = dic["corr"] if dim > 1 else np.eye(dim)
        R = from_upper_triangle(R, dim)
        R_inv = np.linalg.inv(R)

        u = np.zeros((N, dim), dtype=np.float64)
        pdf_x = np.zeros((N, dim), dtype=np.float64)
        score_marginal = np.zeros((N, dim), dtype=np.float64)

        for i in range(dim):
            f = functions[i]
            if not f in ppf_marginals:
                f="normal"
            p = params[i]
            x_col = samples[:, i]

            # ------------------------------------------------------------
            # Normal
            # ------------------------------------------------------------
            if f == "normal":
                mu = p[0]
                sigma = max(abs(p[1]), eps)

                u[:, i] = norm.cdf(
                    x_col,
                    loc=mu,
                    scale=sigma,
                )

                pdf_x[:, i] = norm.pdf(
                    x_col,
                    loc=mu,
                    scale=sigma,
                )

                # d/dx log N(x | mu, sigma^2)
                score_marginal[:, i] = -(x_col - mu) / (sigma ** 2)

            # ------------------------------------------------------------
            # Student-t
            # ------------------------------------------------------------
            elif f == "studentt":
                mu = p[0]
                sigma = max(abs(p[1]), eps)
                df = np.clip(
                    abs(p[2]),
                    2.0 + 1e-2,
                    12.0,
                )

                # sigma is the desired standard deviation.
                # Var(T_df) = df / (df - 2)
                scale = sigma * np.sqrt((df - 2.0) / df)

                z = (x_col - mu) / scale

                u[:, i] = t.cdf(
                    x_col,
                    df=df,
                    loc=mu,
                    scale=scale,
                )

                pdf_x[:, i] = t.pdf(
                    x_col,
                    df=df,
                    loc=mu,
                    scale=scale,
                )

                # d/dx log t(x)
                score_marginal[:, i] = (
                    -(df + 1.0) * z
                    / ((df + z ** 2) * scale)
                )

            # ------------------------------------------------------------
            # Kumaraswamy
            # ------------------------------------------------------------
            elif f == "kumaraswamy":
                mu = p[0]
                sigma = max(abs(p[1]), eps)
                a = abs(p[2]) + eps
                b = abs(p[3]) + eps

                m1, s1 = _kumaraswamy_moments(a, b)

                # X = mu + sigma * (K - m1) / s1
                #
                # Therefore:
                # K = m1 + s1 * (X - mu) / sigma
                # dK/dX = s1 / sigma
                k = m1 + s1 * (x_col - mu) / sigma

                valid = (k > 0.0) & (k < 1.0)

                # Initialize outside-support values.
                u[:, i] = np.nan
                pdf_x[:, i] = 0.0
                score_marginal[:, i] = np.nan

                kv = k[valid]

                # Kumaraswamy CDF
                u[valid, i] = (
                    1.0 - (1.0 - kv ** a) ** b
                )

                # Kumaraswamy PDF
                pdf_k = (
                    a
                    * b
                    * kv ** (a - 1.0)
                    * (1.0 - kv ** a) ** (b - 1.0)
                )

                # f_X(x) = f_K(k) * dk/dx
                pdf_x[valid, i] = pdf_k * s1 / sigma

                # d/dk log f_K(k)
                score_k = (
                    (a - 1.0) / kv
                    - (b - 1.0)
                    * a
                    * kv ** (a - 1.0)
                    / (1.0 - kv ** a)
                )

                # d/dx log f_X(x)
                #
                # log f_X = log f_K(k) + log(s1/sigma)
                # dk/dx = s1/sigma
                score_marginal[valid, i] = (
                    score_k * s1 / sigma
                )

            # ------------------------------------------------------------
            # Exponential
            # ------------------------------------------------------------
            elif f == "exponential":
                mu = p[0]
                sigma = max(abs(p[1]), eps)

                # X = (mu - sigma) + sigma * Exp(1)
                loc = mu - sigma

                u[:, i] = expon.cdf(
                    x_col,
                    loc=loc,
                    scale=sigma,
                )

                pdf_x[:, i] = expon.pdf(
                    x_col,
                    loc=loc,
                    scale=sigma,
                )

                # d/dx log f(x) = -1/sigma
                score_marginal[:, i] = -1.0 / sigma

            # ------------------------------------------------------------
            # Uniform
            # ------------------------------------------------------------
            elif f == "uniform":
                mu = p[0]
                sigma = max(abs(p[1]), eps)

                half = np.sqrt(3.0) * sigma
                low = mu - half
                high = mu + half

                u[:, i] = uniform.cdf(
                    x_col,
                    loc=low,
                    scale=high - low,
                )

                pdf_x[:, i] = uniform.pdf(
                    x_col,
                    loc=low,
                    scale=high - low,
                )

                # log f(x) is constant inside the support
                score_marginal[:, i] = 0.0

            else:
                raise ValueError(
                    f"Unsupported marginal function: {f}"
                )

        # ----------------------------------------------------------------
        # Gaussian copula contribution
        # ----------------------------------------------------------------

        # Avoid norm.ppf(0) and norm.ppf(1).
        u_safe = np.clip(u, eps, 1.0 - eps)

        y = norm.ppf(u_safe)
        phi_y = norm.pdf(y)

        # (R^{-1} - I) y
        y_Rinv = y @ R_inv.T

        # d log c / d u
        #
        # d log c / d y = -(R^{-1} - I)y
        # dy/du = 1 / phi(y)
        dlogc_du = (
            -(y_Rinv - y)
            / np.maximum(phi_y, 1e-300)
        )

        # Chain rule:
        #
        # d/dx log p(x)
        #   = d/dx log c(F_1(x_1), ..., F_d(x_d))
        #     + d/dx log f_i(x_i)
        #
        #   = (d log c / du_i) * f_i(x_i)
        #     + marginal_score_i
        score = (
            dlogc_du * pdf_x
            + score_marginal
        )

        return score


    def prep_optimization(self, dic):
        dic_torch={key:val for key,val in dic.items()}
        dic_torch["params"]=torch.tensor(dic["params"].copy(), dtype=torch.float32, requires_grad=True)
        if "corr" in dic:
            dic_torch["corr"]=torch.tensor(dic["corr"].copy(), dtype=torch.float32, requires_grad=True)# if "corr" in dic else torch.eye(1)
        variables=[dic_torch["params"]]
        if "corr" in dic:variables.append( dic_torch["corr"] )
        #print("dic_torch",dic_torch)
        #print("variables",variables)
        return dic_torch, variables


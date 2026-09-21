import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import torch

#load the pretrained model
from prodigi import load_model, safe_json
key="checkpoint/epoch_550"
draw_batch, encode, decode, read_batch, prior=load_model("final50",f"{key}.pt",f"{key}_base.pt")


#generate some toy data, using the prior we just loaded
dist={'dim': 2,
      'ncomp': 3,
      'weight': np.array([0.6, 0.3,0.1]),
      'mean':
      np.array([[0, 0],
        [-1,1],
        [ 1, 1]]),
      'diag_std':
      np.array([[0.1875, 0.1875],
        [0.25, 0.1875],
        [0.25, 0.1875]]), 'family': 'gmm'}

data=prior.sample(dist, 4000)

print("Original distribution:", json.dumps(safe_json(dist), indent=2))

#normalize the data, save the normalization constants
mean=np.mean(data, axis=0)
std=np.std(data, axis=0)
data=(data-mean)/std

#pad the data to 50dim
data=np.pad(data, ((0,0),(0,50-2)), 'constant', constant_values=0)

#run the data through our model
#our inference api allows for batch inference
batch={"data":np.array([data])}
encoded=encode(batch)
decoded=decode(encoded)
decoded=read_batch(decoded)

reconstructed_dist=decoded["dist"][0]

mmd_mode=reconstructed_dist["family"]=="scm"

#define a loss function. 
if mmd_mode:
    #in case of scms, the NLL loss is not defined. So we have to fall back to an MMD loss. We use the quadratic-time MMD with RBF kernel, which is a standard choice.
    def quadratic_mmd_rbf_torch(X, Y, gamma=1.0):
        """
        Exact PyTorch equivalent of quadratic_mmd_rbf().

        Computes the unbiased quadratic-time MMD^2 with RBF kernel:

            k(x, y) = exp(-gamma * ||x-y||^2)

        X: (n, d)
        Y: (m, d)
        """

        n = X.shape[0]
        m = Y.shape[0]

        dist_xx = torch.sum(
            (X[:, None, :] - X[None, :, :]) ** 2,
            dim=-1
        )

        dist_yy = torch.sum(
            (Y[:, None, :] - Y[None, :, :]) ** 2,
            dim=-1
        )

        dist_xy = torch.sum(
            (X[:, None, :] - Y[None, :, :]) ** 2,
            dim=-1
        )

        K_xx = torch.exp(-gamma * dist_xx)
        K_yy = torch.exp(-gamma * dist_yy)
        K_xy = torch.exp(-gamma * dist_xy)

        mask_xx = ~torch.eye(n, dtype=torch.bool, device=X.device)
        mask_yy = ~torch.eye(m, dtype=torch.bool, device=Y.device)

        xx = K_xx[mask_xx].mean()
        yy = K_yy[mask_yy].mean()
        xy = K_xy.mean()

        return xx + yy - 2.0 * xy


    def loss_mmd(X, Y, dim, gamma=1.0):
        return quadratic_mmd_rbf_torch(
            X,
            Y,
            gamma=gamma / dim
        )
else:
    #In case the prior is a GMM or a Copula, we have access to the density of the distribution at every point and can thus use a NLL loss. While we could also use the same MMD loss as above, the NLL loss is more effective.
    def loss_nll(X, dist):
        logdensity=prior.density_torch(dist, X)
        return -logdensity.mean()


#Select every variable that has gradients. 
dist_torch, variables=prior.prep_optimization(reconstructed_dist)

#define torch optimizer
optimizer=torch.optim.Adam(variables, lr=0.01)
data_torch=torch.tensor(data, dtype=torch.float32)


#Optimize either by mmd or nll
for epoch in tqdm(range(300), desc="Tuning"):
    optimizer.zero_grad()

    if mmd_mode:
        poi=prior.sample_torch(dist_torch, data.shape[0])
        loss=loss_mmd(data_torch[:,:2], poi[:,:2], dim=2)
    else:
        loss=loss_nll(data_torch[:,:2], dist_torch)

    loss.backward()


    # Kill this optimization if any gradient is NaN/Inf.
    if any(
        v.grad is not None
        and not torch.isfinite(v.grad).all()
        for v in variables
    ):
        print("Non-finite gradient detected; stopping optimization.")
        break

    optimizer.step()

#convert variables with gradients back to numpy arrays
reconstructed_dist={key:(val.cpu().detach().numpy() if hasattr(val,"numpy") else val) for key,val in dist_torch.items()}



#undo the normalization
#therefore first make sure that the distribution has exactly mean=0, std=1
reconstructed_dist,_=prior.normalize(reconstructed_dist)
#now set the mean and std to the original values
reconstructed_dist=prior.denormalize(reconstructed_dist, {"mean":mean, "std":std})
print("Found distribution:", json.dumps(safe_json(reconstructed_dist), indent=2))



new_samples=prior.sample(reconstructed_dist, 4000)


plt.figure(figsize=(10, 6))
data_orig=(data[:,:2]*std)+mean
plt.scatter(data_orig[:, 0], data_orig[:, 1], alpha=0.5, label='Original Data', color='blue')
plt.scatter(new_samples[:, 0], new_samples[:, 1], alpha=0.5, label='Reconstructed Samples', color='orange')

plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.savefig("images/finetune.png",dpi=300)

plt.show()






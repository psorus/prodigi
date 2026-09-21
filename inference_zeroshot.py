import matplotlib.pyplot as plt
import numpy as np
import json


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

plt.savefig("images/zeroshot.png",dpi=300)


plt.show()






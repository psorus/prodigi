import numpy as np

def generate_weights(count, alpha=None, minimum_weight=0.01, minimum_delta=0.01):
    if alpha is None:
        alpha=np.exp(np.random.uniform(-2,2))
    free_range=1.0-count*minimum_weight
    free_range-=minimum_delta*(count-1)*(count)/2
    if free_range<0:
        raise ValueError("Cannot generate weights with the given constraints")

    weights=np.random.dirichlet(np.ones(count)*alpha)
    weights=weights*free_range+minimum_weight
    weights=np.sort(weights)#sort ascending
    weights+=minimum_delta*np.arange(count) #add the minimum delta to each weight
    return weights[::-1]


if __name__=="__main__":
    print(generate_weights(10, alpha=None, minimum_weight=0.01, minimum_delta=0.01))



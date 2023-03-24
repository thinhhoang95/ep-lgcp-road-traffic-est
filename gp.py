import numpy as np
from math import exp

def rbf_kernel(x1, x2, l = 1):
    return exp(-1 * ((x1-x2) ** 2) / (2*l))
    
def gram_matrix(xs, var, l):
    gram = var * np.array([[rbf_kernel(x1,x2,l) for x2 in xs] for x1 in xs])
    return gram

def sample_from_GP(xs, mean, var, l):
    # Get the gram matrix
    gram = gram_matrix(xs, var, l)
    mean_vec = np.array([mean for x in xs])
    # Sample from the GP
    ys = np.random.multivariate_normal(mean_vec, gram)
    return ys
import numpy as np
from numpy.linalg import norm
from numpy import dot
def euclidian_distance(x1, x2):
    sq=(x1-x2)**2
    sigma_sq=np.sum(sq)
    euc_dist = np.sqrt(sigma_sq)
    return euc_dist

def manhattan_distance(x1, x2):
    return np.sum((x1-x2))

def cosine_similarity(x1, x2):
    dist = dot(x1, x2) /norm(x1)*norm(x2)
    return dist

def jaccard_distance(x1,x2):
    z=set(x1).intersection(set(x2))
    a=float(len(z))/(len(x1)+len(x2)-len(z))
    return 1-a
        
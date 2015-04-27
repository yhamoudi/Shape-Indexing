import numpy as np
from numpy import linalg
from scipy import linalg as LA

import scipy.sparse.linalg

# Generate the following matrix:
# [[-A.  In   0.   0.  0.]
# [ In  -A.  In    0.  0.]
# [ 0.  In   -A   In   0.]
# [ 0.  0.   In   -A.  In]
# [ 0.  0.    0.  In  -A.]]

# With A the following n*n matrix:

# [[-4.  1.  0.  0.]
#  [ 1. -4.  1.  0.]
#  [ 0.  1. -4.  1.]
#  [ 0.  0.  1. -4.]]

def generate_laplacian_matrix(n, h):
    N = n*n
    a = np.diagflat(-4*np.ones(N), k=0)
    b = np.diagflat(np.ones(N-1), k=1)
    c = np.diagflat(np.ones(N-1), k=-1)
    d = np.diagflat(np.ones(N-n), k=-n)
    e = np.diagflat(np.ones(N-n), k=n)
    return (a+b+c+d+e)/(h*h)

# Given a n*n image, compute the corresponding vector
# Then compute the eigenvalues of the image v
def compute_eigenvalues(img):
    n = img.shape[0]
    img.resize(n*n,1)
    M = generate_laplacian_matrix(n, 3.14159/float(n-1))
    laplacian = M * img
    #w = sparse.linalg.eigs(laplacian, k=6, which='SR', return_eigenvectors=False)
    #w = LA.eig(laplacian)
    w= np.linalg.eigvals(laplacian)
    #w = scipy.sparse.linalg.eigs(laplacian, return_eigenvectors=False)
    w = w[w<-0.001]
    return sorted(-w)


def compute_descriptor(eigenvalues):
    descriptor = [eigenvalues[0]/eigenvalues[1]]
    descriptor.append(eigenvalues[0]/eigenvalues[2])
    descriptor.append(eigenvalues[0]/eigenvalues[3])
    descriptor.append(eigenvalues[1]/eigenvalues[2])
    descriptor.append(eigenvalues[2]/eigenvalues[3])
    return descriptor

def eigenvalues_square(n):
    # Creation of a square img:
    img = np.ones((n, n), dtype=np.float)
    img[:, 0] = np.zeros(n)
    img[:, n-1] = np.zeros(n)
    img[0, :] = np.zeros(n)
    img[n-1, :] = np.zeros(n)

    return compute_eigenvalues(img)

if __name__=="__main__":
    a = eigenvalues_square(5)
    b = eigenvalues_square(15)
    c = eigenvalues_square(50)

    print(a)
    print(b)
    print(c)

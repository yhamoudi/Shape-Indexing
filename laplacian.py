__author__ = 'quentin'
import numpy
from numpy import linalg

# Generate the following matrix:
# [[-A.  In   0.   0.  0.]
# [ In  -A.  In    0.  0.]
# [ 0.  In   -A   In   0.]
# [ 0.  0.   In   -A.  In]
# [ 0.  0.    0.  In  -A.]]

# With A the following n*n matrix:

# [[-4.  1.  1.  0.]
#  [ 1. -4.  1.  1.]
#  [ 1.  1. -4.  1.]
#  [ 0.  1.  1. -4.]]


def generate_laplacian_matrix(n):
    N = n*n
    a = numpy.diagflat(-4*numpy.ones(N), k=0)
    b = numpy.diagflat(numpy.ones(N-1), k=1)
    c = numpy.diagflat(numpy.ones(N-1), k=-1)
    d = numpy.diagflat(numpy.ones(N-n), k=-n)
    e = numpy.diagflat(numpy.ones(N-n), k=n)
    return a+b+c+d+e


# Given a n*n image, compute the corresponding vector
# Then compute the eigenvalues of the image v
def compute_eigenvalues(img):
    n = img.shape[0]
    img.resize(n*n,1)
    M = generate_laplacian_matrix(n)
    laplacian = M * img
    w, vectors = linalg.eig(laplacian)
    return sorted(-w)




n = 10

# Creation of a square img:
img = numpy.ones((n, n), dtype=numpy.float)
img[:, 0] = numpy.zeros(n)
img[:, n-1] = numpy.zeros(n)
img[0, :] = numpy.zeros(n)
img[n-1, :] = numpy.zeros(n)


print(compute_eigenvalues(img))


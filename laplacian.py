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

# [[-4.  1.  0.  0.]
#  [ 1. -4.  1.  0.]
#  [ 0.  1. -4.  1.]
#  [ 0.  0.  1. -4.]]


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
    w = w[w<-0.001]
    return sorted(-w)


def eigenvalues_square(n):
    # Creation of a square img:
    img = numpy.ones((n, n), dtype=numpy.float)
    img[:, 0] = numpy.zeros(n)
    img[:, n-1] = numpy.zeros(n)
    img[0, :] = numpy.zeros(n)
    img[n-1, :] = numpy.zeros(n)

    return compute_eigenvalues(img)


print(eigenvalues_square(5))
print(eigenvalues_square(10))
print(eigenvalues_square(20))
print(eigenvalues_square(40))



import numpy as np

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


def generate_laplacian_matrix(height, weight):
    N = height*weight
    a = np.diagflat(-4*np.ones(N), k=0)
    b = np.diagflat(np.ones(N-1), k=1)
    c = np.diagflat(np.ones(N-1), k=-1)
    d = np.diagflat(np.ones(N-weight), k=-weight)
    e = np.diagflat(np.ones(N-weight), k=weight)
    return (a+b+c+d+e)*(height*weight)


# Given a n*n image, compute the corresponding vector
# Then compute the eigenvalues of the image v
def compute_eigenvalues(img):
    height = img.shape[0]
    weight = img.shape[1]
    img.resize(height*weight,1)
    M = generate_laplacian_matrix(height, weight)
    laplacian = M * img
    w= np.linalg.eigvals(laplacian).real
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

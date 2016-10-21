from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
      Tuning parameter of the RBF kernel

    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset

     lambdas: list
       Eigenvalues

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    # Collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1,n_components+1)]

    return alphas, lambdas




def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

def half_moon_case():

    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, random_state=123)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)

    plt.tight_layout()
    # plt.savefig('./figures/half_moon_1.png', dpi=300)
    plt.show()


    #use new rbf kernel pca
    # alphas = top k eigenvectors (projected samples)
    # lambdas = the corresponding eigenvalues
    alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)


    #Quick test first
    # now we can project one data anywhere we want
    # we assume the 26'th data is a new incoming data.
    x_new = X[25] # the 26 data we want to see the project result
    print(x_new)#original x,y
    '''
    array([ 1.8713,  0.0093])
    '''
    #we peek the answer in our old list.
    #(becasue we need the answer to confirm the project_x() is right or not)
    # because when we meet the real new data, we don't have the real answer,
    #so we try the data in our data set for testing .
    x_proj = alphas[25] # the corresponding original projection
    print(x_proj)
    '''
    array([ 0.0788])
    '''
    # projection of the "new" datapoint
    x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
    print(x_reproj)
    '''
    array([ 0.0788])
    '''
    # x_reproj == x_proj , which means the function project_x() can
    # help us to transform any single data to new space correctlly.

    #plot the data 26th and its proj postion.
    plt.scatter(alphas[y==0, 0], np.zeros((50)),
            color='red', marker='^',alpha=0.5)
    plt.scatter(alphas[y==1, 0], np.zeros((50)),
                color='blue', marker='o', alpha=0.5)
    plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
    plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
    plt.legend(scatterpoints=1)

    plt.tight_layout()
    # plt.savefig('./figures/reproject.png', dpi=300)
    plt.show()

    #ok, you can see x_proj postion is the same with x_reproj
    # so the project_x() working good.


    return 0


def concentric_circles_case():
    #make data
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)

    plt.tight_layout()
    # plt.savefig('./figures/circles_1.png', dpi=300)
    plt.show()

    #use new rbf kernel pca
    # alphas = top k eigenvectors (projected samples)
    # lambdas = the corresponding eigenvalues
    alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)


    #Quick test first
    # now we can project one data anywhere we want
    # we assume the 26'th data is a new incoming data.
    x_new = X[25] # the 26 data we want to see the project result
    print(x_new)#original x,y

    #we peek the answer in our old list.
    #(becasue we need the answer to confirm the project_x() is right or not)
    # because when we meet the real new data, we don't have the real answer,
    #so we try the data in our data set for testing .
    x_proj = alphas[25] # the corresponding original projection
    print(x_proj)

    # projection of the "new" datapoint
    x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
    print(x_reproj)




    plt.scatter(alphas[y==0, 0], np.zeros((500,1)),color='red', marker='^', alpha=0.5)
    plt.scatter(alphas[y==1, 0], np.zeros((500,1)),color='blue', marker='o', alpha=0.5)

    plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
    plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
    plt.legend(scatterpoints=1)

    plt.tight_layout()
    # plt.savefig('./figures/circles_3.png', dpi=300)
    plt.show()

    return 0

def main():


    #Example 1: Separating half-moon shapes
    half_moon_case()

    #Example 2: Separating concentric circles
    print("concentric circles case is not right ?")
    #concentric_circles_case()

    return 0
main()

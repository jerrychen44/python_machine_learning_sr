import matplotlib.pyplot as plt

###########################
# RBF kernel PCA
###########################

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

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
    X_pc = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_components + 1)))

    return X_pc


#################################################


def half_moon_case():

    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=100, random_state=123)
    #X=100 datax2features
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)

    plt.tight_layout()
    # plt.savefig('./figures/half_moon_1.png', dpi=300)
    plt.show()


    #We try original linaer PCA first.
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scikit_pca = PCA(n_components=2)
    #original X is 2 dimetion, so change to 2 pca, alomst the same.
    X_spca = scikit_pca.fit_transform(X)
    print(X_spca.shape)
    # n_components=1, then  X_spca.shape=(100, 1)
    # n_components=2, then  X_spca.shape=(100, 2)

    #plot the result
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))

    #figure left
    ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],
                color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
                color='blue', marker='o', alpha=0.5)

    #figure right
    ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02,
                color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02,
                color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    plt.tight_layout()
    # plt.savefig('./figures/half_moon_2.png', dpi=300)
    plt.show()
    #we found linear PCA can not separate the data in 1 dimension



    #####################
    # Use our home made rbf pca
    #####################
    from matplotlib.ticker import FormatStrFormatter

    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
    #how to chose the gamma? we will choose it automatic later.
    # right now , 15 is because the experience.

    #plot out
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
    ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
                color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
                color='blue', marker='o', alpha=0.5)

    ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02,
                color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,
                color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    plt.tight_layout()
    # plt.savefig('./figures/half_moon_3.png', dpi=300)
    plt.show()
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


    #try origianl linear PCA
    from sklearn.decomposition import PCA
    scikit_pca = PCA(n_components=2)
    X_spca = scikit_pca.fit_transform(X)

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))

    ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],
                color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
                color='blue', marker='o', alpha=0.5)

    ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+0.02,
                color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))-0.02,
                color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    plt.tight_layout()
    # plt.savefig('./figures/circles_2.png', dpi=300)
    plt.show()
    # still can not separate the data

    #################
    # RBF PCA
    ################
    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
    ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
                color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
                color='blue', marker='o', alpha=0.5)

    ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02,
                color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02,
                color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    plt.tight_layout()
    # plt.savefig('./figures/circles_3.png', dpi=300)
    plt.show()

    return 0

def main():


    #Example 1: Separating half-moon shapes
    #half_moon_case()

    #Example 2: Separating concentric circles
    concentric_circles_case()

    return 0
main()

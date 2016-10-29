import numpy as np
import matplotlib.pyplot as plt


def half_moon_case():

    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, random_state=123)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)

    plt.tight_layout()
    # plt.savefig('./figures/half_moon_1.png', dpi=300)
    plt.show()

    ##################
    # Use sklearn rbf PCA
    ###################
    from sklearn.decomposition import KernelPCA

    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_skernpca = scikit_kpca.fit_transform(X)


    #plot out
    plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],
                color='red', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],
                color='blue', marker='o', alpha=0.5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    # plt.savefig('./figures/scikit_kpca.png', dpi=300)
    plt.show()



    #plot out 1 dim
    plt.scatter(X_skernpca[y==0, 0], np.zeros((50,1))+0.02,
                color='red', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], np.zeros((50,1))+0.02,
                color='blue', marker='o', alpha=0.5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    # plt.savefig('./figures/scikit_kpca.png', dpi=300)
    plt.show()



    return 0


def concentric_circles_case():
    #make data
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    print(X)
    print(y)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)

    plt.tight_layout()
    # plt.savefig('./figures/circles_1.png', dpi=300)
    plt.show()

    ##################
    # Use sklearn rbf PCA
    ###################
    from sklearn.decomposition import KernelPCA

    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_skernpca = scikit_kpca.fit_transform(X)


    #plot out 2 dim
    plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],
                color='red', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],
                color='blue', marker='o', alpha=0.5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    # plt.savefig('./figures/scikit_kpca.png', dpi=300)
    plt.show()


    #plot out 1 dim
    plt.scatter(X_skernpca[y==0, 0],np.zeros((500,1)),
                color='red', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], np.zeros((500,1)),
                color='blue', marker='o', alpha=0.5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    # plt.savefig('./figures/scikit_kpca.png', dpi=300)
    plt.show()

    return 0


def main():


    #Example 1: Separating half-moon shapes
    half_moon_case()

    #Example 2: Separating concentric circles
    concentric_circles_case()

    return 0
main()

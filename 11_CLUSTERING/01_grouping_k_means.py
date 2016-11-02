import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import os,sys
import numpy as np
filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
source_folder='source'
print(filepath)
data_csv_path=filepath+'/'+source_folder+'/house.csv'



def sample_data():

    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=150,
                      n_features=2,
                      centers=3,
                      cluster_std=0.5,
                      shuffle=True,
                      random_state=0)


    plt.scatter(X[:,0], X[:,1],  c='white', marker='o', s=50)
    plt.grid()
    plt.tight_layout()
    #plt.savefig('./figures/spheres.png', dpi=300)
    plt.show()
    return X,y

from sklearn.cluster import KMeans
def classic_kmean_and_kmean_plusplus(X,y):
    km = KMeans(n_clusters=3,
                #init='random',
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
    y_km = km.fit_predict(X)

    plt.scatter(X[y_km==0,0],
                X[y_km==0,1],
                s=50,
                c='lightgreen',
                marker='s',
                label='cluster 1')
    plt.scatter(X[y_km==1,0],
                X[y_km==1,1],
                s=50,
                c='orange',
                marker='o',
                label='cluster 2')
    plt.scatter(X[y_km==2,0],
                X[y_km==2,1],
                s=50,
                c='lightblue',
                marker='v',
                label='cluster 3')
    plt.scatter(km.cluster_centers_[:,0],
                km.cluster_centers_[:,1],
                s=250,
                marker='*',
                c='red',
                label='centroids')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plt.savefig('./figures/centroids.png', dpi=300)
    plt.show()
    #return trained model
    return km

def elbow_method_pick_group(km,X,y):
    print('Distortion: %.2f' % km.inertia_)
    '''Distortion: 72.48'''
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X)
        distortions .append(km.inertia_)
    plt.plot(range(1,11), distortions , marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    #plt.savefig('./figures/elbow.png', dpi=300)
    plt.show()

    return 0

from matplotlib import cm
from sklearn.metrics import silhouette_samples
def silhouette_plots(X,y):


    km = KMeans(n_clusters=3,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
    y_km = km.fit_predict(X)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    # plt.savefig('./figures/silhouette.png', dpi=300)
    plt.show()

    return 0

def check_the_bad_example(X,y):
    km = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
    y_km = km.fit_predict(X)

    #plot result
    plt.scatter(X[y_km==0,0],
                X[y_km==0,1],
                s=50,
                c='lightgreen',
                marker='s',
                label='cluster 1')
    plt.scatter(X[y_km==1,0],
                X[y_km==1,1],
                s=50,
                c='orange',
                marker='o',
                label='cluster 2')
    #print the center
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, marker='*', c='red', label='centroids')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plt.savefig('./figures/centroids_bad.png', dpi=300)
    plt.show()



    #silhouette plot
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    # plt.savefig('./figures/silhouette_bad.png', dpi=300)
    plt.show()
    return 0

def main():

    #create sample data
    X,y=sample_data()

    #use classic kmean to take a quick view
    km=classic_kmean_and_kmean_plusplus(X,y)

    ###################
    # Evaluation method 1: elbow fig to group number selection.
    #####################
    #Using the elbow method to find the optimal number of clusters
    elbow_method_pick_group(km,X,y)

    ###################
    # Evaluation method 2: silhouette plots to Quantifying the quality
    # all Clustering algo can use this.
    #####################
    #Quantifying the quality of clustering via silhouette plots
    silhouette_plots(X,y)
    print("you can see the pic that all cluster are away from 0")
    print("so these 3 group is good")

    #Comparison to "bad" clustering:
    print("We force to group by 2 , and check the silhouette again.")
    check_the_bad_example(X,y)
    print("The fig show the cluster 2 with larger height and width and closer to 0, is bad group")
    return 0



main()

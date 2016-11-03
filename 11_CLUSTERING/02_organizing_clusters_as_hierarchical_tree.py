import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sample_data():
    np.random.seed(123)

    variables = ['X', 'Y', 'Z']
    labels = ['ID_0','ID_1','ID_2','ID_3','ID_4']

    X = np.random.random_sample([5,3])*10
    df = pd.DataFrame(X, columns=variables, index=labels)
    print(df)
    '''
          X         Y         Z
    ID_0  6.964692  2.861393  2.268515
    ID_1  5.513148  7.194690  4.231065
    ID_2  9.807642  6.848297  4.809319
    ID_3  3.921175  3.431780  7.290497
    ID_4  4.385722  0.596779  3.980443
    '''
    return df,X,labels

def sample_data2():

    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=150,
                      n_features=2,
                      centers=3,
                      cluster_std=0.5,
                      shuffle=True,
                      random_state=0)

    variables = ['X', 'Y']
    labels = []
    for i in range(len(X)):
        labels.append('ID_'+str(i))

    print(labels)
    plt.scatter(X[:,0], X[:,1],  c='white', marker='o', s=50)
    plt.grid()
    #plt.tight_layout()
    #plt.savefig('./figures/spheres.png', dpi=300)
    plt.show()
    print(X)
    print("The result=%s"%y)

    df = pd.DataFrame(X, columns=variables, index=labels)
    print(df)
    return df,X,labels

def hand_made_hierarchical_clustering(df,labels):
    ############
    # hierarchical clustering : complete linkage
    ############
    #Performing hierarchical clustering on a distance matrix for each data to each data
    from scipy.spatial.distance import pdist,squareform

    row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
    print(row_dist)
    '''
              ID_0      ID_1      ID_2      ID_3      ID_4
    ID_0  0.000000  4.973534  5.516653  5.899885  3.835396
    ID_1  4.973534  0.000000  4.347073  5.104311  6.698233
    ID_2  5.516653  4.347073  0.000000  7.244262  8.316594
    ID_3  5.899885  5.104311  7.244262  0.000000  4.382864
    ID_4  3.835396  6.698233  8.316594  4.382864  0.000000
    '''

    '''
    We can either pass a condensed distance matrix (upper triangular)
    from the pdist function, or we can pass the "original" data array and define the
    'euclidean' metric as function argument n linkage. However, we should nott pass
    the squareform distance matrix, which would yield different distance values although
    the overall clustering could be the same.
    '''

    # 1. incorrect approach: Squareform distance matrix

    from scipy.cluster.hierarchy import linkage

    row_clusters = linkage(row_dist, method='complete', metric='euclidean')
    print(pd.DataFrame(row_clusters,
                 columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
                 index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])]))

    ''' (This result is wrong!!!!)
               row label 1  row label 2   distance  no. of items in clust.
    cluster 1          0.0          4.0   6.521973                     2.0
    cluster 2          1.0          2.0   6.729603                     2.0
    cluster 3          3.0          5.0   8.539247                     3.0
    cluster 4          6.0          7.0  12.444824                     5.0
    '''
    # 2. correct approach: Condensed distance matrix

    row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
    print(pd.DataFrame(row_clusters,
                 columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
                 index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])]))
    '''
               row label 1  row label 2  distance  no. of items in clust.
    cluster 1          0.0          4.0  3.835396                     2.0
    cluster 2          1.0          2.0  4.347073                     2.0
    cluster 3          3.0          5.0  5.899885                     3.0
    cluster 4          6.0          7.0  8.316594                     5.0
    '''

    # 3. correct approach: Input sample matrix (Put original df values.)

    row_clusters = linkage(df.values, method='complete', metric='euclidean')
    print(pd.DataFrame(row_clusters,
                 columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
                 index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])]))
    '''
               row label 1  row label 2  distance  no. of items in clust.
    cluster 1          0.0          4.0  3.835396                     2.0
    cluster 2          1.0          2.0  4.347073                     2.0
    cluster 3          3.0          5.0  5.899885                     3.0
    cluster 4          6.0          7.0  8.316594                     5.0
    '''

    print("linkage matrix is done. (row_clusters)")



    ###############
    # plot out to dendrogram
    ################
    from scipy.cluster.hierarchy import dendrogram

    # make dendrogram black (part 1/2)
    # from scipy.cluster.hierarchy import set_link_color_palette
    # set_link_color_palette(['black'])

    row_dendr = dendrogram(row_clusters,
                           labels=labels,
                           # make dendrogram black (part 2/2)
                           # color_threshold=np.inf
                           )

    #plt.tight_layout()
    plt.ylabel('Euclidean distance')
    #plt.savefig('./figures/dendrogram.png', dpi=300,
    #            bbox_inches='tight')
    plt.show()




    ###########
    # plot out for heatmap
    # Attaching dendrograms to a heat map
    ############
    # plot row dendrogram
    fig = plt.figure(figsize=(8,8))
    axd = fig.add_axes([0.09,0.1,0.2,0.6])
    #row_dendr = dendrogram(row_clusters, orientation='right')
    row_dendr = dendrogram(row_clusters, orientation='left')

    # reorder data with respect to clustering
    df_rowclust = df.ix[row_dendr['leaves'][::-1]]

    axd.set_xticks([])
    axd.set_yticks([])

    # remove axes spines from dendrogram
    for i in axd.spines.values():
            i.set_visible(False)



    # plot heatmap
    axm = fig.add_axes([0.23,0.1,0.6,0.6]) # x-pos, y-pos, width, height
    cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)
    axm.set_xticklabels([''] + list(df_rowclust.columns))
    axm.set_yticklabels([''] + list(df_rowclust.index))

    # plt.savefig('./figures/heatmap.png', dpi=300)
    plt.show()
    print("This heatmap is to mapping with the original df array number.")


    return 0


def scikit_learn_agglomerative_clustering(X,labels):
    from sklearn.cluster import AgglomerativeClustering

    ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
    labels = ac.fit_predict(X)
    print('Cluster labels: %s' % labels)
    ''' if n_clusters=2 Cluster labels: [0 1 1 0 0] '''
    ''' if n_clusters=3 Cluster labels: [1 0 0 2 1] '''
    ''' The same result with the heatmap we made.   '''

    ''' For sample data 2 (boble)
        Cluster labels:
        [1 2 2 2 1 2 2 1 0 2 1 0 0 2 2 0 0 1 0 1 2 1 2 2 0 1 1 2 0 1 0 0 0 0 2 1 1
         1 2 2 0 0 2 1 1 1 0 2 0 2 1 2 2 1 1 0 2 1 0 2 0 0 0 0 2 0 2 1 2 2 2 1 1 2
         1 2 2 0 0 2 1 1 2 2 1 1 1 0 0 1 1 2 1 2 1 2 0 0 1 1 1 1 0 1 1 2 0 2 2 2 0
         2 1 0 2 0 2 2 0 0 2 1 2 2 1 1 0 1 0 0 0 0 1 0 0 0 2 0 1 0 2 2 1 1 0 0 0 0
         1 1]
    '''
    ''' (compare with above Cluster result. (0 <->2))
    The result we already knows =
        [1 0 0 0 1 0 0 1 2 0 1 2 2 0 0 2 2 1 2 1 0 1 0 0 2 1 1 0 2 1 2 2 2 2 0 1 1
         1 0 0 2 2 0 1 1 1 2 0 2 0 1 0 0 1 1 2 0 1 2 0 2 2 2 2 0 2 0 1 0 0 0 1 1 0
         1 0 0 2 2 0 1 1 0 0 1 1 1 2 2 1 1 0 1 0 1 0 2 2 1 1 1 1 2 1 1 0 2 0 0 0 2
         0 1 2 0 2 0 0 2 2 0 1 0 0 1 1 2 1 2 2 2 2 1 2 2 2 0 2 1 2 0 0 1 1 2 2 2 2
         1 1]
     '''
    return 0

def main():
    #create sample_data
    #simple data to explain the algo.
    #df,X,labels=sample_data()

    # try the boble data when we use in k-means
    df,X,labels=sample_data2()
    #Data ready above


    #We Dont have to assiagn the clustering group number
    hand_made_hierarchical_clustering(df,labels)

    #use sklearn version
    scikit_learn_agglomerative_clustering(X,labels)


    return 0
main()

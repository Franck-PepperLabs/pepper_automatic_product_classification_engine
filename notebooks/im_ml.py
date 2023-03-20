import time
import math
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from IPython.display import display
#from sklearn.preprocessing import StandardScaler
#from sklearn import cluster
#from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import preprocessing
from sklearn import cluster
from sklearn import manifold
from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
    davies_bouldin_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pepper_utils import (
    print_file_size,
    get_start_time,
    print_time_perf,
    print_memory_usage
)
from im_prep import (
    get_filenames,
    load_desc_data,
    save_desc_data,
    # get_all_1050_dir,
    # get_sample_8_dir,
    # get_sample_100_dir
)


def get_centers(data, cl_labels):
    """Compute the centers of the clusters in the RFM space.

    Parameters:
    - data (pd.DataFrame): image features data.
    - cl_labels (List[int]): Cluster labels for each observation in `data`.
    """
    data_cl_labeled = pd.concat([
        pd.Series(cl_labels, index=data.index, name='k_cl'),
        data
    ], axis=1)
    cl_means = data_cl_labeled.groupby(by='k_cl').mean()
    return cl_means.values


def kmeans_clustering(data, k, normalize=False):
    km_t = -time.time()
    # Normalize the data
    if normalize:
        data = pd.DataFrame(
            preprocessing.StandardScaler().fit_transform(data),
            index=data.index,
            columns=data.columns
        )

    kmeans = cluster.KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit_predict(data)
    km_t += time.time()
    clu_labels = kmeans.labels_
    clu_centers = get_centers(data, clu_labels)
    return kmeans, clu_labels, clu_centers, km_t



""" k-Means best k
"""

def select_k_with_anova(
    crfm,
    normalize=False,
    k_min=2, k_max=150,
    metric='inertia',
    color='purple',
    verbose=False
):
    """
    Select the optimal number of clusters using ANOVA and the elbow method.
    Parameters
    ----------
    crfm: Pandas DataFrame
        DataFrame with columns 'R', 'F', and 'M' representing
        recency, frequency, and monetary value of customer orders.
    k_min: int, optional (default=2)
        Minimum number of clusters to test.
    k_max: int, optional (default=20)
        Maximum number of clusters to test.
    metric: str, optional (default='inertia')
        Metric to use for the ANOVA. Can be either 'inertia' or 'silhouette'.
    verbose: bool, optional (default=True)
        Whether to print the time taken to fit and predict
        with the KMeans model for each value of k.
    Returns
    -------
    k: int
        Optimal number of clusters.
    """
    # Normalize the data
    if normalize:
        crfm = pd.DataFrame(
            preprocessing.StandardScaler().fit_transform(crfm),
            index=crfm.index,
            columns=crfm.columns
        )

    # Create a list of k values to test
    k_values = list(range(k_min, k_max+1))

    # Initialize lists to store the scores
    anova_scores = []

    # Loop over k values
    for k in k_values:
        # Create a KMeans model with k clusters
        kmeans, clu_labels, _, km_t = kmeans_clustering(crfm, k)
        if verbose:
            print(
                f'Time for kmeans_clustering with k={k} :',
                round(km_t, 3), 's'
            )

        # Calculate the anova score for the current model
        if metric == 'inertia':
            score = kmeans.inertia_
        elif metric == 'silhouette':
            score = silhouette_score(crfm, clu_labels)
        else:
            raise ValueError('Invalid metric')

        # Append the score to the list
        anova_scores.append(score)

    # Plot the scores
    plt.plot(k_values, anova_scores, color=color)
    plt.fill_between(k_values, anova_scores, color=color, alpha=0.2)
    plt.xlabel('Number of clusters')
    plt.xticks(k_values)
    if metric == 'inertia':
        plt.ylabel('Inertia')
    elif metric == 'silhouette':
        plt.ylabel('Silhouette Score')
    norm_status = ', not scaled'
    if normalize:
        norm_status = ', scaled'
    plt.title(f'ANOVA with Elbow Method ({metric}{norm_status})', weight='bold')

    plt.savefig(
        f'../img/ANOVA with Elbow Method_{metric}{norm_status}.png',
        facecolor='white',
        bbox_inches='tight',
        dpi=300   # x 2
    )

    plt.show()


def select_k_with_davies_bouldin(data, normalize=False, k_min=2, k_max=20, k_step=1):
    """Calculate the Davies-Bouldin index for each value of k
    and plot the results as a bar chart.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to cluster.
    k_min, k_max, k_step : int
        The minimum and maximum number of clusters to consider.
    """
    # Normalize the data
    if normalize:
        #crfm[crfm.columns].loc[:, :] 
        data = preprocessing.StandardScaler().fit_transform(data)

    # Create a list of k values to test
    k_values = list(range(k_min, k_max+1, k_step))

    scores = []
    dts = []
    for k in k_values:
        t = time.time()
        kmeans = cluster.KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(data)
        scores.append(davies_bouldin_score(data, kmeans.labels_))
        dts.append(time.time() - t)

    # Sort the scores in descending order
    scores_sorted = sorted(scores)

    # Plot the bar chart
    plt.bar(k_values, scores)
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies-Bouldin index")
    # plt.xticks(k_values)
    plt.ylim(bottom=.5)

    # Mark the three best values of k with red bars
    for i in range(3):
        best_k = scores.index(scores_sorted[i]) * k_step + k_min
        plt.bar(best_k, scores[(best_k-k_min) // k_step], color='green')

    norm_status = ', not scaled'
    if normalize:
        norm_status = ', scaled'
    plt.title(f'Davis-Bouldin index{norm_status}', weight='bold')

    plt.savefig(
        f'../img/Davis-Bouldin index{norm_status}.png',
        facecolor='white',
        bbox_inches='tight',
        dpi=300   # x 2
    )

    plt.show()
    best_ks = [scores.index(score) * k_step + k_min for score in scores_sorted]
    return scores, best_ks, dts


def ml_load_desc_data(dir_solver, sub_dir, data_name):
    file_path = dir_solver(f"{sub_dir}{data_name}.parquet")
    print(data_name, end=' ')
    print_file_size(file_path)
    t = get_start_time()
    features_dir = dir_solver(sub_dir)
    data = load_desc_data(data_name, features_dir)
    kpts = data.iloc[:, :2].values
    descs = data.iloc[:, 2:].values
    print_time_perf(f"{data_name} loaded", f"from {features_dir}", t)
    print("shape:", data.shape)
    print_memory_usage(data)
    return kpts, descs, data


def get_im_descs(data, id):
    return data.loc[id].iloc[:, 2:].values


def get_im_synth_feature(kmeans, im_descs):
    """
    Pour chaque image, on synthétise un descripteur d'image en tant qu'histogramme
        des appartenances des points d'intérêts décrits à tel ou tel cluster.
    """
    clusters = kmeans.predict(im_descs)
    k = len(kmeans.cluster_centers_)
    hist = np.zeros(k, dtype=float)
    for c in clusters:
        hist[c] += 1
    return hist / len(im_descs)


def count_parquet_rows(dataset_path):
    dataset = pq.ParquetDataset(dataset_path)
    n_rows = 0
    for fragment in dataset.fragments:
        n_rows += fragment.count_rows()
    return n_rows

def _locate_dataset(dir_solver, data_name):
    sub_dir = f"_im_descs/{data_name}/"
    dir_path = dir_solver(sub_dir)
    filenames = get_filenames(
        dir_path,
        ext='parquet',
        recursive=False,
        meth='glob'
    )
    filenames = [name[:-len(".parquet")] for name in filenames]
    return sub_dir, dir_path, filenames


def dataset_fragments_n_descs_k(dir_solver, data_name):
    _, dir_path, filenames = _locate_dataset(dir_solver, data_name)
    
    print("Dataset fragments:")
    for filename in filenames:
        print("\t", filename)

    # nombre de descripteurs
    n_descs = count_parquet_rows(dir_path)
    print("n_descs:", n_descs)

    k = int(round(math.sqrt(n_descs), 0))
    print("estimated best k : ", k)

    return filenames, n_descs, k


def train_kmeans(dir_solver, data_name):
    t = time.time()

    print(f"data_name ({data_name})")
    sub_dir = "_im_descs/"
    kpts, descs, data = ml_load_desc_data(dir_solver, sub_dir, data_name)
    display(data.shape)
    display(descs.shape)

    # nombre de descripteurs
    n_descs = data.shape[0]
    print("n_descs:", n_descs)

    k = int(round(math.sqrt(n_descs), 0))
    print("estimated best k : ", k)
    
    kmeans = cluster.KMeans(
        n_clusters=k,
        n_init='auto',
        random_state=42,
        # default heuristic : init_size=3*k
    )
    
    kmeans.fit(descs)

    print_time_perf("KMeans.fit(descs)", None, t)

    return kmeans    


def train_batch_kmeans(dir_solver, data_name, k):
    # détermination des clusters / entraînement incrémental du kmeans
    sub_dir, dir_path, filenames = _locate_dataset(dir_solver, data_name)

    kmeans = cluster.MiniBatchKMeans(
        n_clusters=k,
        n_init='auto',
        random_state=42,
        # default heuristic : init_size=3*k
    )

    for data_name in filenames:
        t = time.time()
        print(f"data_name ({data_name})")
        kpts, descs, data = ml_load_desc_data(dir_solver, sub_dir, data_name)
        display(data.shape)
        kmeans.partial_fit(descs)
        print_time_perf("MiniBatchKMeans.partial_fit(descs)", None, t)

    return kmeans


def get_bag_of_words(kmeans, dir_solver, data_name):
    start_t = time.time()

    print(f"data_name ({data_name})")
    sub_dir = "_im_descs/"
    kpts, descs, data = ml_load_desc_data(dir_solver, sub_dir, data_name)
    display(data.shape)

    im_ids = data.index.unique()
    #im_ids.extend(ids)
    im_features = []
    for i, id in enumerate(im_ids):
        im_descs = get_im_descs(data, id)
        im_feature = get_im_synth_feature(kmeans, im_descs)
        im_features.append(im_feature)

    im_features = np.asarray(im_features)
    
    display(im_features.shape)
    print_time_perf("Building of synthetic features", None, start_t)

    return im_features, im_ids


def get_batch_bag_of_words(kmeans, dir_solver, data_name):
    start_t = time.time()

    sub_dir, dir_path, filenames = _locate_dataset(dir_solver, data_name)

    im_features = []
    im_ids = []
    for data_name in filenames:
        t = time.time()
        print(f"data_name ({data_name})")
        kpts, descs, data = ml_load_desc_data(dir_solver, sub_dir, data_name)
        display(data.shape)
        ids = data.index.unique()
        im_ids.extend(ids)
        for i, id in enumerate(ids):
            im_descs = get_im_descs(data, id)
            im_feature = get_im_synth_feature(kmeans, im_descs)
            im_features.append(im_feature)
        print_time_perf("get_im_synth_feature", data_name, t)

    im_features = np.asarray(im_features)

    display(im_features.shape)
    print_time_perf("Building of synthetic features", None, start_t)

    return im_features, im_ids


def save_im_features(dir_solver, data_name, ids, features):
    features_data = pd.DataFrame(
        features,
        index=ids,
        columns=[str(i) for i in range(features.shape[1])]
    )
    root_dir = dir_solver("_im_feats/")
    data_name = f"{data_name[:-5]}feats"
    save_desc_data(features_data, data_name, root_dir)


""" Dimensionality reduction
"""

def load_features(dir_solver, sub_dir, data_name):
    root_dir = dir_solver(sub_dir)
    features = load_desc_data(data_name, root_dir)
    display(features.shape)
    features.info()
    return features


def load_im_features(dir_solver, data_name):
    sub_dir = "_im_feats/"
    return load_features(dir_solver, sub_dir, data_name)


def load_red_im_features(dir_solver, data_name):
    sub_dir = "_red_im_feats/"
    return load_features(dir_solver, sub_dir, data_name)


""" tSNE
"""

def tsne_reduction(red_im_features, as_dataframe=False):
    n_components = 2   # default: 2
    perplexity = 5.0   # default: 30.0. in [5.0, 50.0] & < n_samples
    n_iter = 2000      # default: 1000
    init = "random"
    
    tsne = manifold.TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        init=init,
        random_state=42
    )
    X_tsne = tsne.fit_transform(red_im_features)

    if not as_dataframe:
        return X_tsne
    else:
        return pd.DataFrame(
            X_tsne,
            columns=[f"tsne_{i}" for i in range(n_components)],
            index=red_im_features.index
        )


def get_tsne_km_clusters(tsne_feats, k=7):
    cls = cluster.KMeans(n_clusters=k, n_init="auto", random_state=42)
    cls.fit(tsne_feats)
    return cls.labels_


def show_tsne(tsne_feats, classes, labels_type):
    tsne_0 = tsne_feats[:, 0]
    tsne_1 = tsne_feats[:, 1]
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=tsne_0, y=tsne_1,
        hue=classes, legend="brief",
        palette=sns.color_palette("tab10", n_colors=7), s=50, alpha=0.6)

    plt.title(f"TSNE against {labels_type}", fontsize=30, pad=35, fontweight="bold")
    plt.xlabel("tsne_0", fontsize=26, fontweight="bold")
    plt.ylabel("tsne_1", fontsize=26, fontweight="bold")
    plt.legend(prop={"size": 14}) 

    plt.show()


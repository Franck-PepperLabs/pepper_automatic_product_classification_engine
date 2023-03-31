from typing import *
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
    # silhouette_samples,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.utils import check_array

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


DirSolver = Callable[[str], str]


def get_centers(data: pd.DataFrame, cl_labels: List[int]) -> np.ndarray:
    r"""Compute the centers of the clusters in the RFM space.

    Parameters:
    -----------
    data: pd.DataFrame
        DataFrame with columns 'R', 'F', and 'M' representing
        recency, frequency, and monetary value of customer orders.
    cl_labels: List[int]
        Cluster labels for each observation in `data`.

    Returns:
    --------
    np.ndarray
        An array of shape (n_clusters, n_features) representing
        the coordinates of the centers of the clusters.
    """
    data_cl_labeled = pd.concat([
        pd.Series(cl_labels, index=data.index, name='k_cl'),
        data
    ], axis=1)
    cl_means = data_cl_labeled.groupby(by='k_cl').mean()
    return cl_means.values


def kmeans_clustering(
    data: pd.DataFrame,
    k: int,
    normalize: bool = False
) -> Tuple[
    cluster.KMeans,
    List[int],
    pd.DataFrame,
    float
]:
    r"""Perform K-means clustering on the input data.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to cluster.
    k : int
        The number of clusters to create.
    normalize : bool, optional
        Whether to normalize the data before clustering,
        by default False.

    Returns
    -------
    Tuple[cluster.KMeans, List[int], pd.DataFrame, float]
        A tuple containing the fitted KMeans model, the
        cluster labels, the cluster centers, and the time
        taken to fit the model and generate the clusters.

    Raises
    ------
    ValueError
        If the specified value of `k` is less than 1 or
        greater than or equal to the number of rows in `data`.

    """
    km_t = -time.time()
    # Normalize the data
    if normalize:
        data = pd.DataFrame(
            preprocessing.StandardScaler().fit_transform(data),
            index=data.index,
            columns=data.columns
        )

    if k < 1 or k >= len(data):
        raise ValueError(
            "k must be greater than 0 "
            "and less than the number of rows in data."
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
    crfm: pd.DataFrame,
    normalize: bool = False,
    k_min: int = 2,
    k_max: int = 150,
    metric: str = 'inertia',
    color: str = 'purple',
    verbose: bool = False,
) -> int:
    r"""Select the optimal number of clusters using the elbow method.
    
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



def select_k_with_davies_bouldin(
    data: pd.DataFrame,
    normalize: bool = False,
    k_min: int = 2,
    k_max: int = 20,
    k_step: int = 1
) -> Tuple[List[float], List[int], List[float]]:
    r"""Calculate the Davies-Bouldin index for each value of k
    and plot the results as a bar chart.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to cluster.
    normalize : bool, optional
        Whether to normalize the data before clustering, by default False.
    k_min : int, optional
        The minimum number of clusters to consider, by default 2.
    k_max : int, optional
        The maximum number of clusters to consider, by default 20.
    k_step : int, optional
        The step size between each number of clusters, by default 1.

    Returns
    -------
    Tuple[List[float], List[int], List[float]]
        A tuple containing the Davies-Bouldin index score for each k value,
        the list of the best k values sorted in ascending order,
        and the time taken to compute the DB index for each k value.
    """
    # Normalize the data
    if normalize:
        # data = preprocessing.StandardScaler().fit_transform(data)
        # TODO : cette modification à tester
        data = pd.DataFrame(
            preprocessing.StandardScaler().fit_transform(data),
            index=data.index,
            columns=data.columns
        )
    
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


def ml_load_desc_data(
    dir_solver: Callable[[str], str],
    sub_dir: str,
    data_name: str
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    r"""Load keypoints and descriptors from a parquet file.

    Parameters
    ----------
    dir_solver : callable
        A function that takes a subdirectory as argument and returns the full
        path to that subdirectory.
    sub_dir : str
        The subdirectory containing the data.
    data_name : str
        The name of the data to load (without file extension).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pd.DataFrame]
        A tuple containing the keypoints as a numpy array, the descriptors as
        a numpy array, and the data as a pandas DataFrame.
    """
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


def get_im_descs(data: pd.DataFrame, id: int) -> np.ndarray:
    """
    Extracts the image descriptors for a given image ID from the dataframe.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe containing the image descriptors.
    id : int
        The ID of the image for which to extract the descriptors.

    Returns:
    --------
    np.ndarray
        The descriptors for the specified image as a numpy array.
    """
    return data.loc[id].iloc[:, 2:].values


def get_im_synth_feature(
    kmeans: cluster.KMeans,
    im_descs: np.ndarray
) -> np.ndarray:
    r"""Synthesizes an image feature descriptor for a set of image descriptors
    using KMeans clustering.

    For each image, an image descriptor is synthesized as a histogram of the
    memberships of the keypoints described to one of the clusters.

    Parameters:
    -----------
    kmeans : KMeans
        The KMeans clustering model.
    im_descs : np.ndarray
        The image descriptors for the images to synthesize the feature
        descriptors.

    Returns:
    --------
    np.ndarray
        The synthesized feature descriptor as a numpy array.
    """
    clusters = kmeans.predict(im_descs)
    k = len(kmeans.cluster_centers_)
    hist = np.zeros(k, dtype=float)
    for c in clusters:
        hist[c] += 1
    return hist / len(im_descs)


def count_parquet_rows(dataset_path: str) -> int:
    r"""Counts the total number of rows in a Parquet dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the Parquet dataset.

    Returns
    -------
    int
        The total number of rows in the dataset.
    """
    dataset = pq.ParquetDataset(dataset_path)
    n_rows = 0
    for fragment in dataset.fragments:
        n_rows += fragment.count_rows()
    return n_rows


def _locate_dataset(
    dir_solver: Callable[[str], str],
    data_name: str
) -> Tuple[str, str, List[str]]:
    r"""Locates a Parquet dataset directory in the specified directory tree.

    Parameters
    ----------
    dir_solver : DirSolver
        The directory solver function.
    data_name : str
        The name of the dataset.

    Returns
    -------
    Tuple[str, str, List[str]]
        A tuple containing the sub-directory path of the dataset, the full path
        to the dataset directory, and a list of the filenames of the dataset
        fragments.
    """
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


def dataset_fragments_n_descs_k(
    dir_solver: DirSolver,
    data_name: str
) -> Tuple[List[str], int, int]:
    r"""Counts the number of descriptors and estimated best k for a Parquet
    dataset.

    Parameters:
    -----------
    dir_solver : DirSolver
        The directory solver function.
    data_name : str
        The name of the dataset.

    Returns:
    --------
    Tuple[List[str], int, int]
        A tuple containing a list of the filenames of the dataset fragments,
        the total number of descriptors in the dataset, and the estimated best
        value of k.
    """
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


def train_kmeans(dir_solver: DirSolver, data_name: str) -> cluster.KMeans:
    r"""Trains a KMeans clustering model for the image descriptors of a given
    dataset.

    Parameters:
    -----------
    dir_solver : DirSolver
        A function that takes a subdirectory name and returns the corresponding
        absolute directory path.
    data_name : str
        The name of the dataset to load the image descriptors from.

    Returns:
    --------
    cluster.KMeans
        The trained KMeans clustering model.
    """
    t = time.time()

    print(f"data_name ({data_name})")
    sub_dir = "_im_descs/"
    kpts, descs, data = ml_load_desc_data(dir_solver, sub_dir, data_name)
    display(data.shape)
    display(descs.shape)

    # Number of descriptors
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


def train_batch_kmeans(
    dir_solver: DirSolver,
    data_name: str,
    k: int
) -> cluster.MiniBatchKMeans:
    """Trains a MiniBatchKMeans model on multiple Parquet datasets, where each
    dataset contains image descriptors. The training is performed incrementally,
    one dataset at a time.

    Parameters:
    -----------
    dir_solver : DirSolver
        The function used to locate directories.
    data_name : str
        The name of the datasets to train on.
    k : int
        The number of clusters to form.

    Returns:
    --------
    MiniBatchKMeans
        The trained MiniBatchKMeans model.
    """
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


# TODO : rename in get_bag_of_features
def get_bag_of_words(
    kmeans: cluster.KMeans,
    dir_solver: DirSolver,
    data_name: str
) -> Tuple[np.ndarray, list]:
    r"""Generates a bag of words representation of a dataset based on a kmeans
    model. Each image is represented as a histogram of the memberships of the
    keypoints described to one of the clusters.

    Parameters
    ----------
    kmeans : KMeans
        The KMeans clustering model.
    dir_solver : DirSolver
        An instance of DirSolver class used to locate the image descriptor
        files.
    data_name : str
        The name of the dataset.

    Returns
    -------
    Tuple[np.ndarray, list]
        The bag of words representation of the dataset as a numpy array and the
        list of the image ids.
    """
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


def get_batch_bag_of_words(
    kmeans: cluster.MiniBatchKMeans,
    dir_solver: DirSolver,
    data_name: str
) -> Tuple[np.ndarray, List[str]]:
    r"""Compute the synthetic features (also known as bag of words) for a
    dataset using MiniBatchKMeans in batch mode.

    Parameters
    ----------
    kmeans : cluster.MiniBatchKMeans
        The trained MiniBatchKMeans object to use for feature encoding.
    dir_solver : DirSolver
        The DirSolver object used to access the dataset.
    data_name : str
        The name of the dataset to compute the features for.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        A tuple containing the synthetic features as a numpy array and a list
        of image ids.
    """
    start_t = time.time()

    sub_dir, _, filenames = _locate_dataset(dir_solver, data_name)

    im_features = []
    im_ids = []
    for data_part_name in filenames:
        t = time.time()
        print(f"data_part_name ({data_part_name})")
        _, _, data = ml_load_desc_data(dir_solver, sub_dir, data_part_name)
        display(data.shape)
        ids = data.index.unique()
        im_ids.extend(ids)
        # for i, id in enumerate(ids):
        # TODO : tester cette modification
        for id in ids:
            im_descs = get_im_descs(data, id)
            im_feature = get_im_synth_feature(kmeans, im_descs)
            im_features.append(im_feature)
        print_time_perf("get_im_synth_feature", data_part_name, t)

    im_features = np.asarray(im_features)

    display(im_features.shape)
    print_time_perf("Building of synthetic features", None, start_t)

    return im_features, im_ids


def save_im_features(
    dir_solver: str,
    data_name: str,
    ids: List[str],
    features: pd.DataFrame
) -> None:
    """Save image features as a pandas DataFrame.

    The features are saved as a DataFrame with the image IDs as the index and
    each feature dimension as a separate column. The resulting DataFrame is
    then saved as a file using the DirSolver object.

    Parameters
    ----------
    dir_solver : str
        The root directory for saving the image features.
    data_name : str
        The name to be given to the saved file. This name will be modified to
        indicate that it is an image feature file.
    ids : List[str]
        A list of image IDs that corresponds to the rows of the features.
    features : pd.DataFrame
        The image features to be saved. The features should be a 2D array-like
        object with the same number of rows as the length of `ids`.

    Returns
    -------
    None
    """
    # Check input TODO : ajout : tester
    features = check_array(features, dtype="numeric", ensure_2d=True)

    # Create DataFrame
    features_data = pd.DataFrame(
        features,
        index=ids,
        columns=[str(i) for i in range(features.shape[1])]
    )

    # Save DataFrame
    root_dir = dir_solver("_im_feats/")
    data_name = f"{data_name[:-5]}feats"
    save_desc_data(features_data, data_name, root_dir)


""" Dimensionality reduction
"""

def load_features(
    dir_solver: DirSolver, sub_dir: str, data_name: str
) -> pd.DataFrame:
    r"""Load the features of the given dataset from disk.

    Parameters
    ----------
    dir_solver : DirSolver
        The DirSolver object used to access the dataset.
    sub_dir : str
        The subdirectory name where the data is stored.
    data_name : str
        The name of the file containing the features.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the features.
    """
    root_dir = dir_solver(sub_dir)
    features = load_desc_data(data_name, root_dir)
    display(features.shape)
    features.info()
    return features


def load_im_features(
    dir_solver: DirSolver, data_name: str
) -> pd.DataFrame:
    r"""Load the image features of the given dataset from disk.

    Parameters
    ----------
    dir_solver : DirSolver
        The DirSolver object used to access the dataset.
    data_name : str
        The name of the file containing the features.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the image features.
    """
    sub_dir = "_im_feats/"
    return load_features(dir_solver, sub_dir, data_name)


def load_red_im_features(
    dir_solver: DirSolver, data_name: str
) -> pd.DataFrame:
    r"""Load the reduced image features of the given dataset from disk.

    Parameters
    ----------
    dir_solver : DirSolver
        The DirSolver object used to access the dataset.
    data_name : str
        The name of the file containing the features.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the reduced image features.
    """
    sub_dir = "_red_im_feats/"
    return load_features(dir_solver, sub_dir, data_name)


""" tSNE
"""

def tsne_reduction(
    red_im_features: pd.DataFrame,
    as_dataframe: bool = False
) -> Union[pd.DataFrame, pd.ndarray]:
    r"""Apply t-distributed Stochastic Neighbor Embedding (t-SNE) to the
    reduced image features in red_im_features and returns the resulting 2D
    embedding. If `as_dataframe` is True, returns the result as a pandas
    DataFrame.
    
    Parameters
    ----------
    red_im_features : pandas.DataFrame
        The reduced image features obtained from applying Principal Component
        Analysis (PCA) to the image features. The DataFrame must be of shape
        (n_samples, n_components), where n_samples is the number of images and
        n_components is the number of principal components.
        
    as_dataframe : bool, default=False
        Whether to return the result as a pandas DataFrame or a numpy array.
        
    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        The 2D embedding of the reduced image features obtained by t-SNE. If
        `as_dataframe` is True, returns a pandas DataFrame with shape
        (n_samples, 2), where n_samples is the number of images. Otherwise,
        returns a numpy array with shape (n_samples, 2).
    """
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


def get_tsne_km_clusters(tsne_feats: np.ndarray, k: int = 7) -> np.ndarray:
    r"""Applies k-means clustering to the 2D embedding of the reduced image
    features obtained by t-SNE.
    
    Parameters
    ----------
    tsne_feats : numpy.ndarray
        The 2D embedding of the reduced image features obtained by t-SNE.
        The array must have shape (n_samples, 2), where n_samples is the number
        of images.
        
    k : int, default=7
        The number of clusters to form.
        
    Returns
    -------
    numpy.ndarray
        An array of shape (n_samples,) containing the cluster labels for each
        image.
    """
    cls = cluster.KMeans(n_clusters=k, n_init="auto", random_state=42)
    cls.fit(tsne_feats)
    return cls.labels_


def show_tsne(
    tsne_feats: np.ndarray, 
    classes: np.ndarray, 
    labels_type: str
) -> None:
    """
    Plot the t-SNE embedding in 2D.

    Parameters
    ----------
    tsne_feats : numpy.ndarray
        The 2D embedding of the reduced image features obtained by t-SNE. It
        should be a numpy array with shape (n_samples, 2), where n_samples is
        the number of images.

    classes : numpy.ndarray
        A numpy array with the class labels for each image. It should have the
        same length as `tsne_feats`.

    labels_type : str
        A string indicating the type of label used for the plot. This is used
        in the plot title.

    Returns
    -------
    None
    """
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

import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
    davies_bouldin_score,
)
import matplotlib.pyplot as plt


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
            StandardScaler().fit_transform(data),
            index=data.index,
            columns=data.columns
        )

    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
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
            StandardScaler().fit_transform(crfm),
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


def select_k_with_davies_bouldin(crfm, normalize=False, k_min=2, k_max=20):
    """Calculate the Davies-Bouldin index for each value of k
    and plot the results as a bar chart.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to cluster.
    k_min, k_max : int
        The minimum and maximum number of clusters to consider.
    """
    # Normalize the data
    if normalize:
        #crfm[crfm.columns].loc[:, :] 
        crfm = StandardScaler().fit_transform(crfm)

    # Create a list of k values to test
    k_values = list(range(k_min, k_max+1))

    scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(crfm)
        scores.append(davies_bouldin_score(crfm, kmeans.labels_))

    # Sort the scores in descending order
    scores_sorted = sorted(scores)

    # Plot the bar chart
    plt.bar(k_values, scores)
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies-Bouldin index")
    plt.xticks(k_values)
    plt.ylim(bottom=.5)

    # Mark the three best values of k with red bars
    for i in range(3):
        best_k = scores.index(scores_sorted[i]) + k_min
        plt.bar(best_k, scores[best_k-k_min], color='green')

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


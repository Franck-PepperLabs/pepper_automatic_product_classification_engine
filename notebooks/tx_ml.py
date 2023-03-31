import os
import timeit
import math
from typing import *

import numpy as np

from scipy.sparse import issparse
from scipy.optimize import linear_sum_assignment as linear_assignment

from IPython.display import display

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cluster, metrics
from sklearn import manifold, decomposition

# Tokenizer
import nltk
from nltk.tokenize import word_tokenize  # sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import tensorflow as tf
import keras
#import tensorflow.keras
from keras import backend as K
#from tensorflow.keras import backend as K

from keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import metrics as kmetrics
#from tensorflow.keras import metrics as kmetrics
from keras.layers import *
#from tensorflow.keras.layers import *
from keras.models import Model
#from tensorflow.keras.models import Model

import tensorflow_hub as hub

import gensim

from transformers import (
    PreTrainedModel,
    AutoTokenizer
)

from pepper_utils import save_and_show

os.environ["TF_KERAS"] = '1'

# import logging
# logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere


""" IO
"""

def load(filename: str) -> pd.DataFrame:
    """
    Loads a CSV file and returns the contents as a pandas DataFrame.

    Parameters:
    filename (str): The name of the file to be loaded.

    Returns:
    pd.DataFrame: The contents of the file as a pandas DataFrame.
    """
    # Set the directory of the file
    dir = r'../dataset/airline_tweets/'
    # Load the CSV file and return its contents as a pandas DataFrame
    return pd.read_csv(dir + filename, encoding='utf-8', index_col='tweet_id')


def save_as(data: pd.DataFrame, filename: str):
    """
    Saves a pandas DataFrame to a CSV file.

    Parameters:
    data (pd.DataFrame): The pandas DataFrame to be saved.
    filename (str): The name of the file to be saved.
    """
    # Set the directory of the file
    dir = r'../dataset/airline_tweets/'
    # Save the DataFrame to the specified file
    data.to_csv(dir + filename, encoding='utf-8')


""" Proprocessing
"""


# def tokenizer_fct(sentence) :
def tokenize_text(sent: str) -> List[str]:
    """Tokenize a sentence into words using the word_tokenize method
    from nltk library.
    
    Args:
    sent: str, the input sentence.

    Returns:
    List[str], the list of tokenized words.
    """
    sentence_clean = (
        sent
        .replace('-', ' ')
        .replace('+', ' ')
        .replace('/', ' ')
        .replace('#', ' ')
    )
    return word_tokenize(sentence_clean)


# def stop_word_filter_fct(list_words) :
def filter_stop_words(word_list: List[str]) -> List[str]:
    """Filter out stop words from a list of words.
    
    Args:
        word_list (List[str]):
            The list of words to filter.

    Returns:
        List[str]:
            The filtered list of words.
    """
    stop_words = (
        stopwords.words('english')
        + ['[', ']', ',', '.', ':', '?', '(', ')']
    )
    return [
        w for w in word_list
        if len(w) > 2 and w not in stop_words
    ]


#def lower_start_fct(list_words) :
def lowercase_and_filter(word_list: List[str]) -> List[str]:
    """Lowercase and filter words that start with @, http or #.
    
    Args:
    word_list: List[str], the list of words to filter.

    Returns:
    List[str], the filtered list of words.
    """
    return [
        w.casefold()
        for w in word_list
        if not (w.startswith("@")
            # or w.startswith("#")
            or w.startswith("http")
        )
    ]


# Lemmatizer (base d'un mot)
# def lemma_fct(list_words) :
def lemmatize_words(word_list: List[str]) -> List[str]:
    """Lemmatize words in a list.
    
    Args:
    word_list: List[str], the list of words to lemmatize.

    Returns:
    List[str], the lemmatized list of words.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in word_list]


#def transform_bow_fct(desc_text) :
def prepare_text_for_bow(text: str) -> str:
    """Prepare text for bag of words analysis (Countvectorizer, Tf-idf, Word2Vec).

    This function performs the following steps:
    1. Tokenize the text
    2. Filter stop words
    3. Lowercase and filter words that start with @, http or #.
    4. Join words into a sentence.
    """
    word_tokens = tokenize_text(text)
    filtered_words = filter_stop_words(word_tokens)
    lowercase_filtered_words = lowercase_and_filter(filtered_words)
    return ' '.join(lowercase_filtered_words)


# Fonction de préparation du texte pour le bag of words avec lemmatization
#def transform_bow_lem_fct(desc_text) :
def prepare_text_for_bow_with_lemmatization(text: str) -> str:
    """Prepare text for bag of words analysis with lemmatization.
    (Countvectorizer, Tf-idf, Word2Vec) 

    This function performs the following steps:
    1. Tokenize the text
    2. Filter stop words
    3. Lowercase and filter words that start with @, http or #.
    4. Lemmatize words
    5. Join words into a sentence.
    """
    word_tokens = tokenize_text(text)
    filtered_words = filter_stop_words(word_tokens)
    lowercase_filtered_words = lowercase_and_filter(filtered_words)
    lemmatized_words = lemmatize_words(lowercase_filtered_words)
    return ' '.join(lemmatized_words)


# def transform_dl_fct(desc_text) :
def prepare_text_for_deep_learning(text: str) -> str:
    """Prepare text for deep learning analysis (USE, BERT, etc.).

    This function performs the following steps:
    1. Tokenize the text
    2. Lowercase and filter words that start with @, http or #.
    3. Join words into a sentence.
    """
    word_tokens = tokenize_text(text)
    lowercase_filtered_words = lowercase_and_filter(word_tokens)
    return ' '.join(lowercase_filtered_words)


def preprocess(raw_tweets: pd.DataFrame) -> None:
    """Preprocess the raw tweets by applying text preparation functions.

    This function performs the following steps:
    1. Apply the `prepare_text_for_bow` function on the 'text' column
        to get the 'sent_bow' column.
    2. Apply the `prepare_text_for_bow_with_lemmatization` function
        on the 'text' column to get the 'sent_bow_lem' column.
    3. Apply the `prepare_text_for_deep_learning` function
        on the 'text' column to get the 'sent_deep_learn' column.

    Args:
    raw_tweets (pandas.DataFrame): The raw tweets dataframe.

    Returns:
    pandas.DataFrame: The preprocessed dataframe with three additional columns:
        'sent_bow', 'sent_bow_lem', 'sent_deep_learn'.
    """
    # Apply the 'prepare_text_for_bow' function on the 'text' column
    # to get the 'sent_bow' column.
    raw_tweets['sent_bow'] = raw_tweets.text.apply(
        prepare_text_for_bow
    )
    # Apply the 'prepare_text_for_bow_with_lemmatization' function
    # on the 'text' column to get the 'sent_bow_lem' column.
    raw_tweets['sent_bow_lem'] = raw_tweets.text.apply(
        prepare_text_for_bow_with_lemmatization
    )
    # Apply the 'prepare_text_for_deep_learning' function
    # on the 'text' column to get the 'sent_deep_learn' column.
    raw_tweets['sent_deep_learn'] = raw_tweets.text.apply(
        prepare_text_for_deep_learning
    )


"""Sampling
"""


def tweets_3k_sample(raw_tweets: pd.DataFrame) -> pd.DataFrame:
    r"""Selects 1500 positive and 1500 negative tweets from the input DataFrame
    using the `sample` method.

    Parameters
    ----------
    raw_tweets : pd.DataFrame
        The input DataFrame containing tweets.

    Returns
    -------
    pd.DataFrame
        The selected 1500 positive and 1500 negative tweets.
    """
    # Concatenate 1500 negative and 1500 positive tweets using the `sample` method
    sample = pd.concat([
        raw_tweets[raw_tweets['airline_sentiment'] == 'negative'].sample(1_500),
        raw_tweets[raw_tweets['airline_sentiment'] == 'positive'].sample(1_500)
    ])
    # Display the shape of the resulting sample DataFrame
    display(sample.shape)
    return sample


def stratified_sample(cats: pd.Series, size: float) -> pd.DataFrame:
    r"""Stratified sampling on a pandas.Series of categories.

    This function performs stratified sampling on a pandas.Series of
    categories, such as the target variable of a classification task.
    It returns a new pandas.DataFrame containing a subset of the original
    data, with the same proportion of samples for each category as in the
    original data.

    Parameters
    ----------
    cats : pandas.Series
        A pandas.Series of categorical labels.
    size : float
        The proportion of samples to keep for each category.

    Returns
    -------
    pd.DataFrame
        A new pandas.DataFrame containing a subset of the original data, with
        the same proportion of samples for each category as in the original
        data.
    """
    tags = cats.unique()
    samples = []
    for tag in tags:
        tag_cats = cats[cats == tag]
        n_total = tag_cats.shape[0]
        n_sample = math.floor(size * n_total)
        sample = tag_cats.sample(n_sample)
        samples.append(sample)

    sample = pd.concat(samples)
    return sample


""" Bagging
"""



def encode_cats(y: pd.Series) -> Tuple[pd.Series, List[str]]:
    r"""Encode the `y` categories into numerical values using pandas.

    The encoding is done by casting the `y` series to a categorical
    type and using the `cat.codes` property to obtain the numerical values.

    Parameters:
    -----------
    y : pd.Series
        The `y` categories.

    Returns:
    --------
    Tuple[pd.Series, List[str]]
    """
    y = y.astype('category')
    cat_labels = y.cat.categories
    cat_codes = y.cat.codes
    return cat_codes, cat_labels


def get_sents_class_labels(
    sents_index: Optional[pd.DataFrame], 
    class_index: pd.Series
) -> Union[pd.Series, pd.DataFrame]:
    r"""Get the class labels for a set of sentences, based on the product
    categories.

    Parameters
    ----------
    sents_index : pd.DataFrame or None
        DataFrame containing the primary keys of products and the sentence IDs,
        or None if the corpus has not been sentence-tokenized. The DataFrame
        must have two columns named "id" and "sent_id".
    class_index : pd.Series
        Series containing the category labels for each product. The index of
        the series must correspond to the primary keys of the products.

    Returns
    -------
    labels : pd.Series or pd.Categorical
        If `sents_index` is not None, a Series containing the category label
        for each sentence, indexed by the sentence ID. If `sents_index` is
        None, a DataFrame containing the category labels for each product,
        indexed by the product ID.
    """
    if sents_index is None:
        return class_index
    else:
        sents_cla_labels = pd.merge(
            sents_index.id,
            class_index,
            left_on='id',
            right_index=True
        )
        sents_cla_labels.set_index('id', inplace=True)
        return sents_cla_labels.cla


def show_sparsity(matrix, contrast='nearest'):
    """Show the sparsity of a matrix.

    Args:
    matrix (np.ndarray or scipy.sparse.csr.csr_matrix): The matrix to visualize.
    contrast (str, optional): The contrast of the image. Default is 'nearest'.

    Returns:
    None
    """
    # Check if matrix is sparse
    if issparse(matrix):
        # Convert the sparse matrix into dense
        matrix = matrix.toarray()

    # Calculate the sparsity index
    sparsity = np.count_nonzero(matrix) / matrix.size

    # Display the sparsity index
    print(f"Sparsity index: {100 * sparsity:.2f}")

    # Set the sparsity (replace values close to 0 with 0)
    # matrix[np.abs(matrix) < 0.1] = 0

    # Use imshow to display the matrix
    plt.imshow(1 - matrix, cmap='gray', vmin=0, vmax=1, interpolation=contrast)
    plt.show()


def tsne_kmeans_ari(
    features: Union[pd.DataFrame, pd.Series, np.ndarray],
    cla_labels: Union[pd.Series, np.ndarray],
    verbosity: int = 1,
    align: bool = True,
    tsne_params: Optional[Dict[str, Any]] = None,
    kmeans_params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    r""" Calculate t-SNE, determine clusters using KMeans, calculate
    Adjusted Rand Index (ARI) between true categories and cluster labels,
    and return t-SNE-transformed features, cluster labels, ARI score,
    total time elapsed, and the t-SNE parameters.

    Parameters
    ----------
    features: array-like, shape (n_samples, n_features)
        The feature matrix of the samples.
    cla_labels: array-like, shape (n_samples,)
        The true classes labels.
    verbosity: int, default=1
        Controls the verbosity of the output:
        - 0: No output
        - 1: Print ARI score only
        - 2: Print ARI score, time elapsed for each step and total time
    align : bool, optional (default=True)
        If True, align the predicted class labels with the ground truth labels
        using the Hungarian algorithm.
    tsne_params: dict, optional
        Additional t-SNE parameters. If not specified, the default parameters
        defined in the function will be used.
    kmeans_params: dict, optional
        Additional KMeans parameters. If not specified, the default parameters
        defined in the function will be used.

    Returns
    -------
    X_tsne: array-like, shape (n_samples, n_components)
        The transformed features using t-SNE.
    labels_: array-like, shape (n_samples,)
        The cluster labels for the samples.
    ARI: float
        The Adjusted Rand Index between true categories and cluster labels.
    total_time: float
        Total time elapsed during the function call.
    tsne_params: dict
        The t-SNE parameters used for the transformation.
    """
    # Fit t-SNE on features
    tsne_time = -timeit.default_timer()

    _tsne_params = {
        'n_components': 2,  # Dimension of the embedded space : fixed
        'perplexity': 30.0,  # [5, 50] Larger datasets usually require a larger perplexity.
        'early_exaggeration': 12.0,  # Not very critical
        'learning_rate': 'auto',  # [10.0, 1000.0],  max(N / early_exaggeration / 4, 50)
        'n_iter': 1000,  # Maximum number of iterations for the optimization (> 250)
        'init': 'random',  # Otherwise 'pca', but not with a sparse input matrix like in this project
        'method': 'barnes_hut',  # Otherwise 'exact' but on small sample (O(N^2) against O(NlogN))
        'random_state': 42,
    }
    _tsne_params.update(tsne_params or {})    
    tsne = manifold.TSNE(**_tsne_params)
    X_tsne = tsne.fit_transform(features)
    tsne_time += timeit.default_timer()
    
    # Fit KMeans on t-SNE transformed features
    kmeans_time = -timeit.default_timer()
    n_cats = len(set(cla_labels))
    _kmeans_params = {
        'n_clusters': n_cats,  # Number of clusters to form as well as the number of centroids to generate
        'n_init': 100,  # Number of time the k-means algorithm will be run with different centroid seeds
        'random_state': 42,
    }
    _kmeans_params.update(kmeans_params or {})    
    cls = cluster.KMeans(**_kmeans_params)
    cls.fit(X_tsne)
    kmeans_time += timeit.default_timer()

    # Calculate Adjusted Rand Index
    ari_time = -timeit.default_timer()
    ari = metrics.adjusted_rand_score(cla_labels, cls.labels_)
    ari_time += timeit.default_timer()
    total_time = tsne_time + kmeans_time + ari_time

    # Print the time elapsed and ARI score
    if verbosity > 1:
        print(f"T-SNE time: {tsne_time:.2f}s")
        print(f"KMeans time: {kmeans_time:.2f}s")
        print(f"ARI time: {ari_time:.2f}s")
        print(f"Total time : {total_time:.2f}s")

        print(f"\nARI : {ari:.4f}")

    clu_labels = cls.labels_
    if align:
        mapping = match_class(clu_labels, cla_labels)
        clu_labels = np.array([mapping[clu] for clu in clu_labels])

    return X_tsne, clu_labels, ari, total_time, tsne.get_params()


def match_class(y_clu, y_cla):
    """
    Find the best matching between true and predicted classes based on
    the size of the intersection between the indices of 
    y_clu and y_cla.

    Args:
    - y_clu (np.ndarray): predicted classes
    - y_cla (np.ndarray): true classes

    Returns:
    - class_mapping (np.ndarray): best match between the predicted
    and true classes based on the size of the intersection 
    between the indices of y_clu and y_cla.

    See:
    https://en.wikipedia.org/wiki/Assignment_problem
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    """
    if isinstance(y_clu, list):
        y_clu = np.array(y_clu)
    if isinstance(y_cla, list):
        y_cla = np.array(y_cla)

    n_clusters = np.unique(y_clu).shape[0]
    n_classes = np.unique(y_cla).shape[0]
    match_matrix = np.zeros((n_clusters, n_classes))

    # print(match_matrix)
    for clu in range(n_clusters):
        for cla in range(n_classes):
            intersection = np.intersect1d(
                np.where(y_clu == clu),
                np.where(y_cla == cla)
            )
            match_matrix[clu, cla] = intersection.shape[0]

    # display(match_matrix)
    return linear_assignment(-match_matrix)[1]


def show_tsne(
    cla_labels: np.ndarray, 
    cla_names: List[str],
    X_tsne: np.ndarray, 
    clu_labels: np.ndarray, 
    results: Optional[dict] = None,
    #what: Optional[str] = None,
    params: Optional[dict] = None,    
) -> None:
    r"""Plot scatter plots of t-SNE representation of a dataset using true
    categories and cluster labels.

    Parameters
    ----------
    cla_labels: numpy.ndarray, shape (n_samples,)
        The numerical categorical label codes of the samples.
    cla_names: list of str, shape (n_labels,)
        The categorical labels.
    X_tsne: numpy.ndarray, shape (n_samples, n_components)
        The t-SNE transformed feature matrix of the samples.
    clu_labels: numpy.ndarray, shape (n_samples,)
        The cluster labels for the samples.
    results: dict, optional (default=None)
        A dictionary containing the results of the clustering algorithm, such
        as the Adjusted Rand Index (ARI) and the R-squared value (R2).
    params: dict, optional (default=None)
        A dictionary containing the parameters used for clustering, t-SNE
        transformation, and data preprocessing.

    Returns
    -------
    None

    """   
    def scat_plot(fig, x, y, pos, labels, cla_names, title):
        ax = fig.add_subplot(pos)
        scatter = ax.scatter(x, y, c=labels, cmap='Set1')
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=list(cla_names), loc="best", title=title
        )
        plt.title(title, pad=10)

    # Find the best matching between true and predicted classes based on ARI
    mapping = match_class(clu_labels, cla_labels)
    _clu_labels = np.array([mapping[clu] for clu in clu_labels])

    x, y = X_tsne[:, 0], X_tsne[:, 1]

    # Initialize the figure
    fig = plt.figure(figsize=(15, 7))

    # Create a scatter plot of true categories
    scat_plot(fig, x, y, 121, cla_labels, cla_names, "True classes")
    
    # Create a scatter plot of cluster labels
    scat_plot(fig, x, y, 122, _clu_labels, cla_names, "Clusters")

    fig.subplots_adjust(top=0.8)
    params = params if params else {}
    kmeans_params = params.get("kmeans", {})
    tsne_params = params.get("tsne", {})
    corpus_params = params.get("corpus", {})
    extractor_params = params.get("extractor", {})

    corpus_name = corpus_params.pop('name', "")
    extractor_name = extractor_params.pop('name', "")

    title = f"Corpus “{corpus_name}” / {extractor_name}"
    plt.suptitle(title, fontsize=15, y=.98, x=0.15, weight="bold", ha='left') #

    def dict_str(d):
        return ", ".join([f"{k}={v}" for k, v in d.items()])

    def details_str(obj_params):
        return f": {dict_str(obj_params)}" if obj_params else ""

    params_cartouche = "\n".join([
        f"kMeans{details_str(kmeans_params)}",
        f"tSNE{details_str(tsne_params)}",
        f"Corpus preproc.{details_str(corpus_params)}",
        f"{extractor_name}{details_str(extractor_params)}"
    ])
    pa_bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.55, .98, params_cartouche, bbox=pa_bbox, ha='left', va='top')

    if results:
        results_cartouche = "\n".join([f"{k}: {v}" for k, v in results.items()])
        pe_bbox = dict(boxstyle='round', facecolor='green', alpha=0.5)
        fig.text(0.85, .98, results_cartouche, bbox=pe_bbox, ha='left', va='top')

    save_and_show(f"tx_tsne_{corpus_name}_{extractor_name}", sub_dir="tx_tsne")


""" Word2Vec
"""


def gesim_tokenize(
    corpus: pd.Series,
    deacc: Optional[bool] = False,
    min_len: Optional[int] = 2,
    max_len: Optional[int] = 15
) -> pd.Series:
    r"""Convert `corpus` into a series of lists of lowercase tokens 
    ignoring tokens that are too short or too long.

    Uses gensim's `simple_preprocess` function applying it to each document
    in `corpus`.

    Parameters
    ----------
    doc : pd.Series of str
        Input documents.
    deacc : bool, optional
        Remove accent marks from tokens using deaccent()
    min_len : int, optional
        Minimum length of token (inclusive). Shorter tokens are discarded.
    max_len : int, optional
        Maximum length of token in result (inclusive). Longer tokens are
        discarded.

    Returns
    -------
        The tokens extracted from each document in `corpus`.
    """
    def simple_preprocess(doc):
        return gensim.utils.simple_preprocess(doc, deacc, min_len, max_len)

    return corpus.apply(simple_preprocess)



def old_gesim_simple_preprocess(
    bow_lem: Union[list, pd.Series]
) -> List[List[str]]:
    """
    DEPRECATED use `gesim_tokenize` instead

    This function preprocesses a list of texts (bow_lem) by applying gensim's
    `simple_preprocess` function to each element in the list.
    The function returns a list of lists of preprocessed words.

    Parameters:
    bow_lem (list): A list of strings.
    
    Returns:
    list of lists of preprocessed words:
        A list where each element is a list of words preprocessed
        by gensim's simple_preprocess function.
    """
    return [
        gensim.utils.simple_preprocess(text)
        for text in bow_lem
    ]


def get_w2v_hyperparams(w2v_model: gensim.models.Word2Vec) -> dict:
    r"""Return a dictionary of hyperparameters with their current values for
    the given Word2Vec model.

    Parameters
    ----------
    w2v_model : gensim.models.Word2Vec
        The trained Word2Vec model to get hyperparameters from.

    Returns
    -------
    dict
        A dictionary containing the hyperparameters of the model and their
        current values. The keys of the dictionary correspond to the
        hyperparameters and the values correspond to their respective values in
        the model.

    Notes
    -----
    The hyperparameters that are not present in the model's current instance
    dictionary are not included in the returned dictionary.

    """
    """excluded_keys = [
        'load', 'wv', 'cum_table', 'lifecycle_events', 'syn1neg',
        'total_train_time', 'random', 'min_alpha_yet_reached',
        'corpus_count', 'corpus_total_words', 'max_final_vocab',
        'null_word', 'raw_vocab', 'layer1_size', 'comment',
        'effective_min_count'
    ]"""
    params_keys = [
        'sg', 'vector_size', 'min_count', 'sample', 'window',
        'cbow_mean', 'hs', 'negative', 'ns_exponent', 'alpha',
        'min_alpha', 'epochs', 'sorted_vocab', 'max_vocab_size',
        'seed', 'workers', # 'batch_words', 'trim_rule', 'callbacks'
    ]
    map_keys = {
        'sg': 'cbow', 'min_count': 'min_df', 'sample': 'hf_thres',
        'epochs': 'n_iter', 'seed': 'random_state'
    }
    map_values = {
        'sg': lambda x: not bool(x), 'cbow_mean': bool, 'hs': bool
    }
    params = w2v_model.__dict__
    return {
        map_keys.get(key, key):
        map_values.get(key, lambda x: x)(params[key])
        for key in params_keys
    }


def create_w2v_model(
    tok_corpus,

    cbow=True,  # Training algorithm: True for CBOW (Continuous Bag Of Words).
                # Otherwise Skip-Gram

    vector_size=100,   # Dimensionality of the word space
                       # Recommended : large corpus [50, 300], small corpus [100, 1000]

    min_df=5,     # Ignores all words with total frequency lower than this.
                  # Recommended value : large corpus [1, 5], small corpus [5, 20]
    hf_thres=1e-3,  # [0.0, 1.0] The threshold for configuring which higher-frequency words
                    # are randomly downsampled (probability of sampling a word with df > df_min)
                    # Recommanded value : [1e-5, 1e-3] => np.logspace(-5, -3, num=10)

    window=5,  # [2, 15] Context window : maximum distance between the current
               # and predicted word within a sentence.
               # Recommended value is 10 for skip-gram and 5 for CBOW
               # [2, 7] for CBOW, [8, 12] for Skip-Gram

    cbow_mean=True,  # If False, use the sum of the context word vectors.
                     # If True, use the mean (generaly a best choice).
                     # Only applies when cbow is used.

    hs=False,  # If True, hierarchical softmax will be used for model training (no negative sampling).
               # If False, and `negative` is non-zero, negative sampling will be used.

    negative=5,  # [0, 20] If > 0, negative sampling will be used.
                 # The int for negative specifies how many "noise words"
                 # should be drawn (usually between 5-20).
                 # Default is 5. If set to 0, no negative samping is used.
                 # small corpus [0, 5], large corpus [5, 20]

    ns_exponent=0.75,  # [0.0, 1.0] The exponent used to shape the negative sampling distribution.
                       # A value of 1.0 samples exactly in proportion to the frequencies,
                       # 0.0 samples all words equally, while a value of less than 1.0
                       # skews towards low-frequency words.

    # Learning rate
    alpha=0.03,        # The initial learning rate.
    min_alpha=0.0007,  # Learning rate will linearly drop to `min_alpha` as training progresses.
                       # small corpus [5e-3, 1e-2] x [5e-4, 1e-3]
                       # middle corpus [1e-2, 5e-2] x [5e-4, 1e-3]
                       # large corpus [5e-2, 1e-1] x [5e-4, 1e-3]
    n_iter=5,  # Number of iterations (epochs) over the corpus.
               # small [5, 10], middle [10, 20], large [20, 100]

    sorted_vocab=True,  # If True, sort the vocabulary by descending frequency before assigning word indices.
                        # If False can be semanticaly performant but costly.
                        # If False, max_vocab_size must be specified (not None)
    max_vocab_size=None,  # Limit RAM during vocabulary building; if there are more unique words than this,
                          # then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
                          # Set to None for no limit.

    random_state=42,  # Seed for the random number generator.
    compute_loss=False,  # If True, compute and report the training loss.
    # batch_words=10000,  # Target size (in words) for batches of examples passed to worker threads.
    # trim_rule=None,
    # callbacks=(),
    workers=1,  # Use these many worker threads to train the model (=faster training with multicore machines).
) -> gensim.models.Word2Vec:
    return gensim.models.Word2Vec(
        tok_corpus,
        # Algo choice
        sg=1-cbow,
        # Word space dim
        vector_size=vector_size,
        # Freq thresholding
        min_count=min_df,
        sample=hf_thres,
        # Context window size
        window=window,
        # Option for CBOW mean against sum
        cbow_mean=0+cbow_mean,
        # Activate the hierarchical softmax (=> no negative sampling)
        hs=0+hs,
        # Negative sampling
        negative=negative,
        ns_exponent=ns_exponent,
        # Learning rate
        alpha=alpha,
        min_alpha=min_alpha,
        epochs=n_iter,
        # Vocabulary sorting (or not)
        sorted_vocab=0+sorted_vocab,
        max_vocab_size=max_vocab_size,
        # More
        seed=random_state,
        compute_loss=compute_loss,
        workers=workers,
    )


def keras_num_and_pad(
    tok_corpus: pd.Series,
    maxlen: int = None
) -> Tuple[Tokenizer, np.ndarray]:
    r"""This function numerizes and pads a series of texts (`tok_corpus`)
    to a specified maximum length (maxlen).

    Parameters
    ----------
    tok_corpus : list
        A list of strings.
    maxlen (int, optional):
        The maximum length of the tokenized texts after padding.
        Default is None.

    Returns
    -------
    tokenizer (Keras Tokenizer object):
        The fitted Keras token
    """
    # print("Fit Tokenizer ...")
    keras_tokenizer = Tokenizer()
    keras_tokenizer.fit_on_texts(tok_corpus)

    # Transform each sequence of words to a sequence of integers
    num_corpus = keras_tokenizer.texts_to_sequences(tok_corpus)

    # Pad sequences of variable length to a fixed size 'maxlen'
    # with padding after the end of the sequence.
    padded_num_corpus = keras.utils.pad_sequences(
        num_corpus,
        maxlen=maxlen,
        padding='post'
    ) 

    # Adding 1 to account for padding token (code 0)
    #num_words = len(keras_tokenizer.word_index) + 1  
    #print(f"Number of unique words: {num_words}")
    return keras_tokenizer, padded_num_corpus


def get_embedding_matrix(
    w2v_model: gensim.models.Word2Vec,
    keras_tokenizer: Tokenizer
) -> np.ndarray:
    """
    Create the word embedding matrix using a gensim word2vec model
    and a Keras Tokenizer.

    Parameters
    ----------
    w2v_model: gensim.models.Word2Vec
        The pre-trained word2vec model.
    keras_tokenizer: Tokenizer
        The Keras Tokenizer used to tokenize the input text.

    Returns
    -------
    embedding_matrix: np.ndarray
        The embedding matrix with shape (vocab_size, vector_size).
    """
    # Create the embedding matrix
    # print("Create Embedding matrix ...")

    # Get the vectors from the word2vec model
    word_vectors = w2v_model.wv
    # old : w2v_vocab = word_vectors.index_to_key
    vector_size = w2v_model.vector_size

    # Get the word index from the tokenizer
    keras_word_index = keras_tokenizer.word_index

    # Adding 1 to account for padding token (code 0)
    vocab_size = len(keras_word_index) + 1

    # Initialize the embedding matrix
    embed_mx = np.zeros((vocab_size, vector_size))

    # Get the words that are common between keras_word_index and w2v_vocab
    common_words = (
        set(keras_word_index.keys())
        & set(word_vectors.index_to_key)
    )

    # Populate the embedding matrix
    for word, id in keras_word_index.items():
        if word in common_words:
            embed_mx[id] = word_vectors[word]
    
    # Calculate the word embedding rate
    verbose = 0
    if verbose > 0:
        word_rate = len(common_words) / len(keras_word_index)
        print(f"Word embedding rate : {word_rate:.2f}")
        print(f"Embedding matrix shape: ", embed_mx.shape)

    return embed_mx


def get_keras_w2v_embedding_model(
    padded_num_corpus: np.ndarray,
    w2v_model: gensim.models.Word2Vec,
    keras_tokenizer: Tokenizer,
    embedding_matrix: np.ndarray,
    #maxlen: int = 24
) -> Model:
    r"""Builds a Keras model using a pre-trained Word2Vec embedding matrix
    
    Parameters
    ----------
    padded_num_corpus : np.ndarray
        The processed text sequences after tokenization and padding
    w2v_model : gensim.models.Word2Vec
        The pre-trained Word2Vec model
    keras_tokenizer : Tokenizer
        The tokenizer used to pre-process the text
    embedding_matrix : np.ndarray
        The pre-trained embedding matrix generated from the Word2Vec model
    # maxlen : int, optional
        The maximum length of the padded text sequences, by default 24
    
    Returns
    -------
    Model
        The Keras model for text classification
    """
    # Get the size of the Word2Vec embedding
    vector_size = w2v_model.vector_size

    # Get the word index from the tokenizer
    word_index = keras_tokenizer.word_index
    vocab_size = len(word_index) + 1   # Adding 1 to account for padding token

    # Build the input layer for the model
    # bad : input_layer = Input(shape=(len(padded_num_corpus), maxlen), dtype='float64')
    # good : input_layer = Input(shape=padded_num_corpus.shape, dtype='float64')

    # Build the word input layer for the embedding layer
    num_corpus_width = padded_num_corpus.shape[1]
    word_input = Input(shape=(num_corpus_width,), dtype='float64')  #? (input_layer)

    # Build the embedding layer using the pre-trained embedding matrix
    word_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[embedding_matrix],
        input_length=num_corpus_width
    )(word_input)

    # Use global average pooling to get a fixed length representation of each
    # text sequence
    word_vec = GlobalAveragePooling1D()(word_embedding)

    # Build the final model using the input and output layers
    model = Model([word_input], word_vec)

    # Print a summary of the model architecture
    verbose = 0
    if verbose > 0:
        model.summary()

    return model


""" BERT
"""


def old_encode_sentences_with_bert(
    corpus: pd.Series,
    bert_tokenizer: AutoTokenizer,
    max_length: int
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ]:
    r"""Encode a list of strings into BERT-compatible inputs.

    Parameters
    ----------
    sents : List[str]
        List of strings to encode.
    bert_tokenizer : transformers.PreTrainedTokenizer
        BERT tokenizer.
    max_length : int
        Maximum length of the encoded inputs.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]
        Tuple of the encoded input ids, token type ids, attention mask, 
        and list of tuples containing input ids, token type ids, and attention mask.
    input_ids : np.ndarray
        Encoded input ids.
    token_type_ids : np.ndarray
        Encoded token type ids.
    attention_mask : np.ndarray
        Encoded attention mask.
    bert_inp_tot : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of tuples containing input ids, token type ids, and attention mask.
    """
    input_ids = []
    token_type_ids = []
    attention_mask = []
    bert_inp_tot = []

    # Encode each string in the input list
    for doc in corpus:
        bert_inp = bert_tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True, 
            return_token_type_ids=True,
            truncation=True,
            return_tensors="tf"
        )
    
        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((
            bert_inp['input_ids'][0], 
            bert_inp['token_type_ids'][0], 
            bert_inp['attention_mask'][0]
        ))

    # Convert lists to numpy arrays
    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)
    
    # Return the encoded inputs
    return input_ids, token_type_ids, attention_mask, bert_inp_tot


def encode_sentences_with_bert(
    corpus: pd.Series,
    max_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Encode a list of strings into BERT-compatible inputs.

    Parameters
    ----------
    corpus : pd.Series
        The preprocessed sentences as a series of strings to be transformed
        into embeddings.
    max_length : int
        Maximum length of the encoded inputs.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of the encoded input ids, token type ids, attention mask.
    """
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    bert_inputs = bert_tokenizer.batch_encode_plus(
        corpus.tolist(),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="tf"
    )

    input_ids = bert_inputs['input_ids']
    token_type_ids = bert_inputs['token_type_ids']
    attention_mask = bert_inputs['attention_mask']

    """ne sert à rien : bert_inp_tot = [
        (input_ids[i], token_type_ids[i], attention_mask[i])
        for i in range(len(corpus))
    ]"""

    # stack the input arrays
    input_ids = np.stack(input_ids)
    token_type_ids = np.stack(token_type_ids)
    attention_mask = np.stack(attention_mask)

    # Return the encoded inputs
    return input_ids, token_type_ids, attention_mask  #, bert_inp_tot


"""
Dans la plupart des cas, les modèles BERT pré-entraînés sont optimisés
pour gérer des entrées de longueur variable,
ce qui signifie qu'ils peuvent gérer des entrées de taille différente
dans un seul appel du modèle plutôt que dans plusieurs appels séparés.
Par conséquent, dans la plupart des cas,
il n'est pas nécessaire de diviser les entrées en batches
pour les envoyer au modèle. La boucle de batch a été retirée
pour simplifier le code et l'améliorer en termes de performance.
"""
def extract_bert_sentence_embeddings(
    model: Union[PreTrainedModel, tf.train.Checkpoint],
    model_type: str,
    sents: List[str],
    max_length: int,
    batch_size: int,
    mode: str = 'HF'
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute sentence embeddings using a pre-trained BERT model.

    Parameters
    ----------
    model : Union[transformers.PreTrainedModel, TFCheckpoint]
        Pre-trained BERT model from either HuggingFace's Transformers or
        Tensorflow Hub.
    model_type : str
        Name of the BERT model, such as "bert-base-uncased".
    sents : List[str]
        List of input sentences to be embedded.
    max_length : int
        Maximum length of the input sentences after tokenization.
    batch_size : int
        Batch size to be used for prediction.
    mode : str, optional
        Mode of prediction, either "HF" for HuggingFace or "TFhub" for
        Tensorflow Hub.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of sentence embeddings and the concatenated hidden states
        of the BERT model.
    """
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    (
        input_ids,
        token_type_ids,
        attention_mask,
        bert_inp_tot
    ) = encode_sentences_with_bert(
        sents, bert_tokenizer, max_length
    )

    # Bert HuggingFace
    if mode == 'HF':
        outputs = model.predict(
            [input_ids, attention_mask, token_type_ids],
            batch_size=batch_size
        )
        last_hidden_states = outputs.last_hidden_state

    # Bert Tensorflow Hub
    elif mode == 'TFhub':
        text_preprocessed = {
            "input_word_ids": input_ids,
            "input_mask": attention_mask,
            "input_type_ids": token_type_ids,
        }
        outputs = model(text_preprocessed)
        last_hidden_states = outputs["sequence_output"]

    bert_features = np.array(last_hidden_states).mean(axis=1)

    return bert_features, bert_inp_tot


""" USE
"""


def extract_use_sentence_embeddings(
    sents: List[str],
    batch_size: int = 10
) -> np.ndarray:
    r"""Extract Universal Sentence Encoder (USE) embeddings for a list of
    sentences.

    Parameters:
    -----------
    sents : list of str
        List of input sentences.
    batch_size : int, optional (default=10)
        Batch size for encoding sentences.

    Returns:
    --------
    use_features : np.ndarray of shape (n_samples, embedding_dim)
        The USE embeddings for each sentence.
    """
    t = -timeit.default_timer()
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    features = np.concatenate([
        embed(sents[k:k+batch_size])
        for k in range(0, len(sents), batch_size)
    ])
    t += timeit.default_timer()
    print(f"USE time: {t:.2f}s")
    return features

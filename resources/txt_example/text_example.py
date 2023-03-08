import os
import timeit
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
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import tensorflow_hub as hub

import gensim

from transformers import (
    PreTrainedModel,
    AutoTokenizer
)

os.environ["TF_KERAS"] = '1'

# import logging
# logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere


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


def old_3k_sample(raw_tweets: pd.DataFrame) -> pd.DataFrame:
    """
    Selects 1500 positive and 1500 negative tweets from the input DataFrame.

    Parameters:
    raw_tweets (pd.DataFrame): The input DataFrame containing tweets.

    Returns:
    pd.DataFrame: The selected 1500 positive and 1500 negative tweets.
    """
    # Concatenate 1500 negative and 1500 positive tweets
    sample = pd.concat(
        (
            raw_tweets[raw_tweets['airline_sentiment'] == 'negative'].iloc[0:1500],
            raw_tweets[raw_tweets['airline_sentiment'] == 'positive'].iloc[0:1500]
        ),
        axis=0
    )
    # Display the shape of the resulting sample DataFrame
    display(sample.shape)
    return sample


def tweets_3k_sample(raw_tweets: pd.DataFrame) -> pd.DataFrame:
    """
    Selects 1500 positive and 1500 negative tweets from the input DataFrame
    using the `sample` method.

    Parameters:
    raw_tweets (pd.DataFrame): The input DataFrame containing tweets.

    Returns:
    pd.DataFrame: The selected 1500 positive and 1500 negative tweets.
    """
    # Concatenate 1500 negative and 1500 positive tweets using the `sample` method
    sample = pd.concat([
        raw_tweets[raw_tweets['airline_sentiment'] == 'negative'].sample(1_500),
        raw_tweets[raw_tweets['airline_sentiment'] == 'positive'].sample(1_500)
    ])
    display(sample.shape)
    return sample


# def tokenizer_fct(sentence) :
def tokenize_text(sent: str) -> List[str]:
    """Tokenize a sentence into words
    using the word_tokenize method from nltk library.
    
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
    word_list: List[str], the list of words to filter.

    Returns:
    List[str], the filtered list of words.
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


def old_encode_cats(sentiment: pd.Series) -> Tuple[pd.Series, List[str]]:
    """Encode the sentiment categories into numerical values.

    DEPRECATED use `encode_cats` instead

    The encoding is done by assigning 0 to the first category in the list,
    1 to the second category, and so on.

    Args:
    sentiment (pandas.Series): The sentiment categories.

    Returns:
    Tuple[pd.Series, List[str]]: The encoded sentiment categories as
        a series of numerical values and the corresponding category labels.
    """
    l_cat = list(set(sentiment))
    y_cat_num = [
        (1-l_cat.index(sentiment.iloc[i]))
        for i in range(len(sentiment))
    ]
    return y_cat_num, l_cat


def encode_cats(sentiment: pd.Series) -> Tuple[pd.Series, List[str]]:
    """Encode the sentiment categories into numerical values using pandas.

    The encoding is done by casting the sentiment series to a categorical
    type and using the `cat.codes` property to obtain the numerical values.

    Args:
    sentiment (pandas.Series): The sentiment categories.

    Returns:
    Tuple[pd.Series, List[str]]: The encoded sentiment categories as
        a series of numerical values and the corresponding category labels.
    """
    sentiment = sentiment.astype('category')
    cat_labels = sentiment.cat.categories
    cat_codes = sentiment.cat.codes
    return cat_codes, cat_labels


def count_words(raw_sent: pd.Series) -> pd.Series:
    """Count the number of words in a pandas Series of strings.

    Args:
    raw_sent (pandas.Series): The pandas Series of strings.

    Returns:
    pandas.Series: The pandas Series with the count of words in each string.
    """
    return len(word_tokenize(raw_sent))


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


# Calcul Tsne, détermination des clusters et calcul ARI entre vrais catégorie et n° de clusters
def tsne_kmeans_ari(
    features: Union[pd.DataFrame, pd.Series, np.ndarray],
    cat_codes: Union[pd.Series, np.ndarray],
    cat_labels: Union[pd.Index, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate TSNE, determine clusters and calculate ARI
    between true categories and cluster labels.

    sql

    Parameters
    ----------
    features: array-like, shape (n_samples, n_features)
        The feature matrix of the samples.
    cat_codes: array-like, shape (n_samples,)
        The categorical label codes.
    cat_labels: list, shape (n_labels,)
        The categorical labels.

    Returns
    -------
    X_tsne: array-like, shape (n_samples, n_components)
        The transformed features using t-SNE.
    labels_: array-like, shape (n_samples,)
        The cluster labels for the samples.
    ARI: float
        The Adjusted Rand Index between true categories and cluster labels.
    """
    # Fit t-SNE on features
    tsne_time = -timeit.default_timer()
    tsne = manifold.TSNE(
        # Number of dimensions to keep in the reduced space
        n_components=2,
        # The perplexity is related to the number of nearest neighbors
        # that is used in other manifold learning algorithms
        perplexity=30,
        # n_iter : int, optional (default=1000)
        # Maximum number of iterations for the optimization.
        # This is used to control the optimization time.
        # The optimization will stop when either this number of iterations
        # is reached or the error tolerance is satisfied.
        n_iter=2000,
        # init : string or np.ndarray, optional (default="random")
        # Initialization of the embedding.
        # Possible options are "random" for random initialization,
        # or a numpy array with shape (n_samples, n_components)
        # with precomputed initialization.
        init='random',
        # learning_rate : float, optional (default=200.0)
        # The learning rate to use in the optimization process.
        # The learning rate controls the speed of convergence
        # and can affect the final result.
        # A lower learning rate may result in a more accurate result,
        # but will take longer to converge.
        learning_rate=200,
        random_state=42
    )
    X_tsne = tsne.fit_transform(features)
    tsne_time += timeit.default_timer()
    
    # Fit KMeans on t-SNE transformed features
    kmeans_time = -timeit.default_timer()
    num_labels = len(cat_labels)
    cls = cluster.KMeans(
        # Number of clusters to form as well
        # as the number of centroids to generate
        n_clusters=num_labels,
        # Number of time the k-means algorithm will be run
        # with different centroid seeds
        n_init=100,
        random_state=42
    )
    cls.fit(X_tsne)
    kmeans_time += timeit.default_timer()

    # Calculate Adjusted Rand Index
    ari_time = -timeit.default_timer()
    ari = metrics.adjusted_rand_score(cat_codes, cls.labels_)
    ari_time += timeit.default_timer()

    # Print the time elapsed and ARI score
    print(f"T-SNE time: {tsne_time:.2f}s")
    print(f"KMeans time: {kmeans_time:.2f}s")
    print(f"ARI time: {ari_time:.2f}s")
    print(f"Total time : {round(tsne_time + kmeans_time + ari_time, 2)}s")

    print(f"\nARI : {round(ari, 4)}")

    return X_tsne, cls.labels_, ari


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
    cat_codes: np.ndarray, 
    cat_labels: List[str],
    X_tsne: np.ndarray, 
    clu_labels: np.ndarray, 
    ARI: float
) -> None:
    """
    Plot scatter plots of t-SNE representation of tweets using true categories and cluster labels.

    Parameters
    ----------
    cat_codes: numpy.ndarray, shape (n_samples,)
        The numerical categorical label codes of the samples.
    cat_labels: list of str, shape (n_labels,)
        The categorical labels.
    X_tsne: numpy.ndarray, shape (n_samples, n_components)
        The t-SNE transformed feature matrix of the samples.
    clu_labels: numpy.ndarray, shape (n_samples,)
        The cluster labels for the samples.
    ARI: float
        The Adjusted Rand Index between true categories and cluster labels.

    Returns
    -------
    None

    """
    # Find the best matching between true and predicted classes based on ARI
    mapping = match_class(clu_labels, cat_codes)
    _clu_labels = np.array([mapping[clu] for clu in clu_labels])

    # Initialize the figure
    fig = plt.figure(figsize=(15, 6))
    
    # Create a scatter plot of true categories
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cat_codes, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=list(cat_labels), loc="best", title="Categories")
    plt.title('Representation of tweets by actual categories')
    
    # Create a scatter plot of cluster labels
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=_clu_labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=list(cat_labels), loc="best", title="Clusters")
    plt.title('Representation of tweets by clusters')
    
    # Show the plot
    plt.show()
    # Print the ARI
    print("ARI : ", ARI)


""" Word2Vec
"""


def gesim_simple_preprocess(
    bow_lem: Union[list, pd.Series]
) -> List[List[str]]:
    """
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


""" Création et entraînement du modèle Word2Vec
"""

"""
w2v_size = 300
w2v_window = 5
w2v_min_count = 1
w2v_epochs = 100
maxlen = 24
"""

def fit_word2vec(
    sents: List[List[str]],
    w2v_size: int = 300,
    w2v_window: int = 5,
    w2v_min_count: int = 1,
    w2v_epochs: int = 100
) -> gensim.models.Word2Vec:
    """
    This function trains a Word2Vec model on a list of texts (sents).
    
    Parameters:
    sents (list): A list of lists of words.
    w2v_size (int, optional):
        The size of the word embeddings to be learned by the model.
        Default is 300.
    w2v_window (int, optional):
        The maximum distance between the current and predicted word
        within a sentence. Default is 5.
    w2v_min_count (int, optional):
        The minimum frequency of words in the vocabulary.
        Words that have a frequency below this are ignored. Default is 1.
    w2v_epochs (int, optional):
        The number of epochs to train the model. Default is 100.
    
    Returns:
    w2v_model: The trained Word2Vec model.
    """
    print("Build & train Word2Vec model ...")
    w2v_model = gensim.models.Word2Vec(
        min_count=w2v_min_count,
        window=w2v_window,
        vector_size=w2v_size,
        seed=42,
        workers=1
    )
    # workers=multiprocessing.cpu_count())
    w2v_model.build_vocab(sents)
    w2v_model.train(
        sents,
        total_examples=w2v_model.corpus_count,
        epochs=w2v_epochs
    )
    model_vectors = w2v_model.wv
    w2v_words = model_vectors.index_to_key
    print(f"Vocabulary size: {len(w2v_words)}")
    print("Word2Vec trained")
    return w2v_model


def fit_keras_tokenizer(
    sents: List[List[str]],
    maxlen: int = 24
) -> Tuple[Tokenizer, np.ndarray]:
    """
    This function tokenizes and pads a list of texts (sents)
    to a specified maximum length (maxlen).

    Parameters:
    sents (list): A list of strings.
    maxlen (int, optional):
        The maximum length of the tokenized texts after padding. Default is 24.

    Returns:
    tokenizer (Keras Tokenizer object): The fitted Keras token
    """
    print("Fit Tokenizer ...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sents)

    # Pad sequences of variable length to a fixed size 'maxlen'
    # with padding after the end of the sequence.
    x_sentences = pad_sequences(
        tokenizer.texts_to_sequences(sents),
        maxlen=maxlen,
        padding='post'
    ) 
                                
    num_words = len(tokenizer.word_index) + 1  # Adding 1 to account for padding token
    print(f"Number of unique words: {num_words}")
    return tokenizer, x_sentences


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
        The embedding matrix with shape (vocab_size, w2v_size).
    """
    # Create the embedding matrix
    print("Create Embedding matrix ...")

    # Get the vectors from the word2vec model
    model_vectors = w2v_model.wv
    w2v_words = model_vectors.index_to_key
    w2v_size = w2v_model.vector_size

    # Get the word index from the tokenizer
    word_index = keras_tokenizer.word_index
    vocab_size = len(word_index) + 1   # Adding 1 to account for padding token

    # Initialize the embedding matrix
    embedding_matrix = np.zeros((vocab_size, w2v_size))

    # TODO : on peut faire mieux avec une comprehension, non ?

    # Populate the embedding matrix
    j = 0
    for word, idx in word_index.items():
        if word in w2v_words:
            embedding_vector = model_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
                j += 1
    
    # Calculate the word embedding rate
    word_rate = np.round(j / len(word_index), 4)
    print(f"Word embedding rate : ", word_rate)
    print(f"Embedding matrix: ", embedding_matrix.shape)

    return embedding_matrix


def get_embedding_model(
    x_sents: np.ndarray,
    w2v_model: gensim.models.Word2Vec,
    keras_tokenizer: Tokenizer,
    embedding_matrix: np.ndarray,
    maxlen: int = 24
) -> Model:
    """
    Builds a Keras model using a pre-trained Word2Vec embedding matrix
    
    Parameters
    ----------
    x_sents : np.ndarray
        The processed text sequences after tokenization and padding
    w2v_model : gensim.models.Word2Vec
        The pre-trained Word2Vec model
    keras_tokenizer : Tokenizer
        The tokenizer used to pre-process the text
    embedding_matrix : np.ndarray
        The pre-trained embedding matrix generated from the Word2Vec model
    maxlen : int, optional
        The maximum length of the padded text sequences, by default 24
    
    Returns
    -------
    Model
        The Keras model for text classification
    """
    # Get the size of the Word2Vec embedding
    w2v_size = w2v_model.vector_size

    # Get the word index from the tokenizer
    word_index = keras_tokenizer.word_index
    vocab_size = len(word_index) + 1   # Adding 1 to account for padding token

    # Build the input layer for the model
    input = Input(
        shape=(len(x_sents), maxlen),
        dtype='float64'
    )

    # Build the word input layer for the embedding layer
    word_input = Input(
        shape=(maxlen,),
        dtype='float64'
    )

    # Build the embedding layer using the pre-trained embedding matrix
    word_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=w2v_size,
        weights=[embedding_matrix],
        input_length=maxlen
    )(word_input)

    # Use global average pooling
    # to get a fixed length representation of each text sequence
    word_vec = GlobalAveragePooling1D()(word_embedding)

    # Build the final model using the input and output layers
    model = Model([word_input], word_vec)

    # Print a summary of the model architecture
    model.summary()

    return model


def encode_sentences_with_bert(
    sents: List[str],
    bert_tokenizer,
    max_length: int
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ]:
    """
    Encode a list of strings into BERT-compatible inputs
    
    Parameters:
    sents (List[str]): list of strings to encode
    bert_tokenizer: BERT tokenizer
    max_length (int): maximum length of the encoded inputs
    
    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    input_ids (np.ndarray): encoded input ids
    token_type_ids (np.ndarray): encoded token type ids
    attention_mask (np.ndarray): encoded attention mask
    bert_inp_tot (List[Tuple[np.ndarray, np.ndarray, np.ndarray]]): list of tuples containing input_ids, token_type_ids, attention_mask
    """
    input_ids = []
    token_type_ids = []
    attention_mask = []
    bert_inp_tot = []

    # Encode each string in the input list
    for sent in sents:
        bert_inp = bert_tokenizer.encode_plus(
            sent,
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
    """
    Compute sentence embeddings using a pre-trained BERT model.

    :param model:
        Pre-trained BERT model from either HuggingFace's Transformers
        or Tensorflow Hub
    :type model: Union[transformers.PreTrainedModel, TFCheckpoint]
    :param model_type: String specifying the name of the BERT model,
        such as "bert-base-uncased"
    :type model_type: str
    :param sents: List of input sentences to be embedded
    :type sentences: List[str]
    :param max_length:
        Maximum length of the input sentences after tokenization
    :type max_length: int
    :param batch_size: Batch size to be used for prediction
    :type batch_size: int
    :param mode: Mode of prediction, either "HF" for HuggingFace
        or "TFhub" for Tensorflow Hub
    :type mode: str, optional
    :return: Tuple of sentence embeddings
        and the concatenated hidden states of the BERT model
    :rtype: Tuple[np.ndarray, np.ndarray]
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
    if mode == 'HF':  # Bert HuggingFace
        outputs = model.predict(
            [input_ids, attention_mask, token_type_ids],
            batch_size=batch_size
        )
        last_hidden_states = outputs.last_hidden_state

    if mode == 'TFhub':  # Bert Tensorflow Hub
        text_preprocessed = {
            "input_word_ids": input_ids,
            "input_mask": attention_mask,
            "input_type_ids": token_type_ids,
        }
        outputs = model(text_preprocessed)
        last_hidden_states = outputs["sequence_output"]

    bert_features = np.array(last_hidden_states).mean(axis=1)
    return bert_features, bert_inp_tot


def extract_use_sentence_embeddings(
    sents: List[str],
    batch_size: int = 10
) -> np.ndarray:
    """
    Extract Universal Sentence Encoder (USE) embeddings for a list of sentences.

    Args:
    - sents: List of input sentences.
    - batch_size: Batch size for encoding sentences.

    Returns:
    - use_features: 2D np.ndarray with the USE embeddings for each sentence.

    """
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    use_features = None
    t = -timeit.default_timer()
    for step in range(len(sents) // batch_size) :
        idx = step * batch_size
        feat = embed(sents[idx:idx+batch_size])

        if step == 0:
            use_features = feat
        else :
            use_features = np.concatenate((use_features, feat))

    t += timeit.default_timer()
    print(f"USE time: {t:.2f}s")
    return use_features
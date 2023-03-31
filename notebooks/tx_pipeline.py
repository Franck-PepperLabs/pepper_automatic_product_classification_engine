from typing import *
import os
import time
import timeit

import pandas as pd
import numpy as np

from nltk import word_tokenize

from IPython.display import display

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.base import BaseEstimator

from pepper_utils import (
    print_title,
    print_subtitle,
    print_subsubtitle,
    bold, pretty_timedelta_str,
    save_and_show,
)

from flipkart_utils import (
    get_raw_data,
    get_class_labels,
    get_class_label_name_map
)

from tx_prep import (
    preprocess_corpus,
    word_tokenize_corpus,
    filter_product_names,
    filter_product_descriptions,
    lens, display_lens_dist, show_lens_dist,
)

from tx_ml import (
    get_sents_class_labels, match_class,
    tsne_kmeans_ari, show_tsne,
)

from scoring import *
from comb_pred import combine_predictions

# tx_ml Word2Vec :
from tx_ml import (
    # gesim_simple_preprocess, DEPRECATED
    gesim_tokenize,
    create_w2v_model,
    get_w2v_hyperparams,
    keras_num_and_pad,
    get_embedding_matrix,
    get_keras_w2v_embedding_model
)

# tx_ml BERT :
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from keras.preprocessing.text import FullTokenizer

#from tranformers.tokenization import FullTokenizer
#from bert.tokenization import FullTokenizer

from tx_ml import (
    encode_sentences_with_bert,
    extract_bert_sentence_embeddings    # features extraction
)

# tx_ml USE :
from tx_ml import extract_use_sentence_embeddings  # features extraction


def tx_ml_preprocess(
    name: str,
    sent_tokenize=False,
    verbosity: int = 0,
    tokenize_func: Optional[Callable[[str], List[str]]] = word_tokenize,
    return_sent_index: bool = False
) -> Tuple[List[str], List[int]]:
    r"""Preprocess text data by applying tokenization and filtering based on
    the text category.

    Parameters
    ----------
    name : str
        The name of the text category. Should be one of
        ['product_name', 'description'].
    sent_tokenize : bool, optional
        Whether to split the text into sentences before preprocessing,
        by default False.
    tokenize_func : Optional[Callable[[str], List[str]]], optional
        The function used to tokenize each string in the Series. 
        If not provided, the function `word_tokenize` from the NLTK package
        will be used.
    return_sent_index : bool, optional
        Whether to return the index of the preprocessed sentences,
        by default False.

    Returns
    -------
    Tuple[List[str], List[str]] or Tuple[List[str], List[str], pd.DataFrame]
        The preprocessed sentences and their corresponding class labels.
        If `return_sent_index` is True, also returns the index of the
        preprocessed sentences as a pandas DataFrame.
    """
    if verbosity > 0:
        print_title(f"Preprocessing of corpus `{name}`")

    discard_words_function = None
    if name == 'product_name':
        discard_words_function = filter_product_names
    elif name == 'description':
        discard_words_function = filter_product_descriptions

    sents, sents_index = preprocess_corpus(
        corpus=get_raw_data(name),
        sent_tokenize=sent_tokenize,
        tokenize_func=tokenize_func,
        discard_words_function=discard_words_function
    )
    cla_labels = get_sents_class_labels(sents_index, get_class_labels())

    if verbosity > 0:
        print("n_sentences:", len(sents))

        tok_sents = word_tokenize_corpus(sents, tokenize_func)
        sent_lens = lens(tok_sents)
        display_lens_dist(sent_lens)
        show_lens_dist(sent_lens, unit='word', log_scale=(True, True))

    if return_sent_index:
        return sents, cla_labels, sents_index
    else:
        return sents, cla_labels


def tl(verbosity: int, title: str) -> None:
    r"""Print the title if the verbosity level is greater than 0.

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    title : str
        The title to print.
    """
    if verbosity > 0:
        print_title(title)


def stl(verbosity: int, subtitle: str) -> None:
    r"""Print the subtitle if the verbosity level is greater than 1.

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    subtitle : str
        The subtitle to print.
    """
    if verbosity > 1:
        print_subtitle(subtitle)


def sstl(verbosity: int, subsubtitle: str) -> None:
    r"""Print the sub-subtitle if the verbosity level is greater than 2.

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    subsubtitle : str
        The sub-subtitle to print.
    """
    if verbosity > 2:
        print_subsubtitle(subsubtitle)



def display_model_params(verbosity: int, extractor: BaseEstimator) -> None:
    r"""Display the parameters of a given feature extractor if the verbosity
    level is greater than 2.

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    extractor : BaseEstimator
        The feature extractor object.
    """
    if verbosity > 2:
        display(extractor)
        print("params:")
        display(extractor.get_params())


def display_features_index(verbosity: int, extractor: BaseEstimator) -> None:
    r"""Display the number and names of the features of a given feature
    extractor if the verbosity level is greater than 2.

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    extractor : BaseEstimator
        The feature extractor object.
    """
    if verbosity > 2:
        feat_names = extractor.get_feature_names_out()
        print("n_features:", len(feat_names))
        display(feat_names)


def display_time(verbosity: bool, what: str, dt: float) -> None:
    if verbosity > 1:
        print(f"{bold((what + ' time').title())}: {pretty_timedelta_str(dt)}")


def create_model(
    verbose: int,
    m_class: Type,
    m_params: dict,
    m_params_update: dict = None
) -> Tuple:
    r"""Create a model object with given class and parameters, and print the
    model parameters if verbosity is greater than 2.

    Parameters
    ----------
    verbose : int
        The verbosity level.
    m_class : Type
        The class of the model to create.
    m_params : dict
        The parameters to create the model with.
    m_params_update : dict, optional
        The additional parameters to update the model parameters with, by
        default None.

    Returns
    -------
    Tuple
        A tuple of the created model object and the start time.
    """
    sstl(verbose, f"Create the {m_class.__name__} object")
    _start_t = -timeit.default_timer()
    m_params.update(m_params_update or {})
    model = m_class(**m_params)
    display_model_params(verbose, model)
    return model, _start_t


class PipelineParams(NamedTuple):
    extractor_class: Type
    reductor_class: Type
    classifier_class: Type
    corpus_params: dict
    extractor_params: dict
    reductor_params: dict
    classifier_params: dict


def pack_pipeline_params(
    ex_class: Type, rd_class: Type, cl_class: Type,
    cp_params: dict, ex_params: dict, rd_params: dict, cl_params: dict,
) -> PipelineParams:
    r"""Pack the pipeline parameters into a PipelineParams named tuple.

    Parameters
    ----------
    ex_class : Type
        The class of the extractor.
    rd_class : Type
        The class of the reducer.
    cl_class : Type
        The class of the classifier.
    cp_params : dict
        The corpus parameters.
    ex_params : dict
        The extractor parameters.
    rd_params : dict
        The reducer parameters.
    cl_params : dict
        The classifier parameters.

    Returns
    -------
    PipelineParams
        The pipeline parameters.

    """
    ex_params['name'] = ex_class.__name__
    rd_params['name'] = rd_class.__name__
    cl_params['name'] = cl_class.__name__
    return PipelineParams(
        extractor_class=ex_class,
        reductor_class=rd_class,
        classifier_class=cl_class,
        corpus_params=cp_params,
        extractor_params=ex_params,
        reductor_params=rd_params,
        classifier_params=cl_params
    )


class PipelineDims(NamedTuple):
    n_docs: int
    n_sents: int
    n_feats: int
    n_rd_feats: int


def pack_pipeline_dims(
    n_docs: int,
    n_sents: int,
    n_feats: int,
    n_rd_feats: int
) -> PipelineDims:
    r"""Pack the pipeline dimensions into a PipelineDims named tuple.

    Parameters
    ----------
    n_docs : int
        The number of documents.
    n_sents : int
        The number of sentences.
    n_feats : int
        The number of features.
    n_rd_feats : int
        The number of reduced features.

    Returns
    -------
    PipelineDims
        The pipeline dimensions.
    """
    return PipelineDims(
        n_docs=n_docs,
        n_sents=n_sents,
        n_feats=n_feats,
        n_rd_feats=n_rd_feats,
    )


class PipelineLabels(NamedTuple):
    cla_labels: pd.Series
    clu_labels: np.ndarray
    comb_cla_labels: pd.Series
    comb_clu_labels: np.ndarray


def pack_pipeline_labels(
    cla_labels: pd.Series, clu_labels: np.ndarray,
    comb_cla_labels: pd.Series, comb_clu_labels: np.ndarray,
) -> PipelineLabels:
    r"""Pack the pipeline labels into a PipelineLabels named tuple.

    Parameters
    ----------
    cla_labels : pd.Series
        The classification labels.
    clu_labels : np.ndarray
        The clustering labels.
    comb_cla_labels : pd.Series
        The combined classification labels.
    comb_clu_labels : np.ndarray
        The combined clustering labels.

    Returns
    -------
    PipelineLabels
        The pipeline labels.

    """
    return PipelineLabels(
        cla_labels=cla_labels,
        clu_labels=clu_labels,
        comb_cla_labels=comb_cla_labels,
        comb_clu_labels=comb_clu_labels,
    )


class PipelineScores(NamedTuple):
    ari: float
    comb_ari: float
    # Not relevant with classification
    # r2: float
    # comb_r2: float


def pack_pipeline_scores(
    ari: float, comb_ari: float,
    # r2: float, comb_r2: float  # Not relevant with classification
) -> PipelineScores:
    r"""Pack pipeline scores into a PipelineScores named tuple.

    Parameters
    ----------
    ari : float
        The Adjusted Rand Index score.
    comb_ari : float
        The combined Adjusted Rand Index score.
    
    # Not relevant with classification
    r2 : float
        The R squared score.
    comb_r2 : float
        The combined R squared score.

    Returns
    -------
    PipelineScores
        The pipeline scores.
    """
    # Not relevant with classification, r2=r2, comb_r2=comb_r2)
    return PipelineScores(ari=ari, comb_ari=comb_ari) 


class PipelineTimes(NamedTuple):
    extractor_time: float
    reductor_time: float
    classifier_time: float
    show_time: float
    total_time: float


def pack_pipeline_times(
    ex_t: float, rd_t: float, cl_t: float, sh_t: float, tt_t: float
) -> PipelineTimes:
    r"""Pack pipeline times into a PipelineTimes named tuple.

    Parameters
    ----------
    ex_t : float
        The time taken for the features extraction.
    rd_t : float
        The time taken for the features reduction.
    cl_t : float
        The time taken for the samples classification.
    sh_t : float
        The time taken for plotting and reporting the results.
    tt_t : float
        The total time taken.

    Returns
    -------
    PipelineTimes
        The pipeline times.
    """
    return PipelineTimes(
        extractor_time=ex_t,
        reductor_time=rd_t,
        classifier_time=cl_t,
        show_time=sh_t,
        total_time=tt_t
    )


class PipelineModels(NamedTuple):
    extractor: object
    reductor: object
    classifier: object


def pack_pipeline_models(
    ex: object, rd: object, cl: object
) -> PipelineModels:
    r"""Pack pipeline models into a PipelineModels named tuple.

    Parameters
    ----------
    ex : object
        The feature extractor model.
    rd : object
        The feature reductor model.
    cl : object
        The classifier model.

    Returns
    -------
    PipelineModels
        The pipeline models.
    """
    return PipelineModels(extractor=ex, reductor=rd, classifier=cl) 


# Portage de l'ex show_tsne
def show_classif_results(
    verbosity: int, rd_feats: np.ndarray,
    labels: PipelineLabels, scores: PipelineScores, params: PipelineParams
) -> None:
    r"""Show the results of the classification.

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    rd_feats : np.ndarray
        The reduced features.
    labels : PipelineLabels
        The labels.
    scores : PipelineScores
        The scores.
    params : PipelineParams
        The parameters.
    """
    if verbosity <= 0:
        return

    def scat_plot(fig, x, y, pos, labels, cla_names, title):
        ax = fig.add_subplot(pos)
        scatter = ax.scatter(x, y, c=labels, cmap='Set1')
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=list(cla_names), loc="best", title=title
        )
        plt.title(title, pad=10)

    # Find the best matching between true and predicted classes based on ARI
    #mapping = match_class(clu_labels, cla_labels)
    #_clu_labels = np.array([mapping[clu] for clu in clu_labels])
    cla_names = list(get_class_label_name_map().values())
    cla_labels = labels.cla_labels
    clu_labels = labels.clu_labels

    x, y = rd_feats[:, 0], rd_feats[:, 1]

    # Initialize the figure
    fig = plt.figure(figsize=(15, 7))

    # Create a scatter plot of true categories
    scat_plot(fig, x, y, 121, cla_labels, cla_names, "Ground true classes")
    
    # Create a scatter plot of cluster labels
    scat_plot(fig, x, y, 122, clu_labels, cla_names, "Predicted classes")

    fig.subplots_adjust(top=0.85, bottom=.1)
    params = params._asdict() if params else {}
    corpus_params = params.get("corpus_params", {})
    extractor_params = params.get("extractor_params", {})
    reductor_params = params.get("reductor_params", {})
    classifier_params = params.get("classifier_params", {})

    corpus_name = corpus_params.pop('name', "")
    extractor_name = extractor_params.pop('name', "")
    reductor_name = reductor_params.pop('name', "")
    classifier_name = classifier_params.pop('name', "")

    title = f"Corpus “{corpus_name}” ⇒ {extractor_name}"
    title += f" ⇒ {reductor_name} ⇒ {classifier_name}"
    plt.suptitle(title, fontsize=15, weight="bold")  #, y=.98, x=0.15, ha='left') #

    def dict_str(d):
        pairs = [
            f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
            for k, v in d.items()
        ]
        if len(pairs) > 10:
            pairs.insert(10, "..\n    ..")
        return ", ".join(pairs)

    def details_str(obj_params):
        return f": {dict_str(obj_params)}" if obj_params else ""

    params_cartouche = "\n".join([
        f"Corpus preproc.{details_str(corpus_params)}",
        f"{extractor_name}{details_str(extractor_params)}",
        f"{reductor_name}{details_str(reductor_params)}",
        f"{classifier_name}{details_str(classifier_params)}",
    ])
    pa_bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(
        .15, .05,
        params_cartouche, bbox=pa_bbox,
        ha='left', va='top', fontsize=8
    )

    if scores:
        scores_cartouche = "\n".join(
            [f"{k}: {v:.4e}" for k, v in scores._asdict().items()]
        )
        pe_bbox = dict(boxstyle='round', facecolor='green', alpha=0.5)
        fig.text(0.9, .98, scores_cartouche, bbox=pe_bbox, ha='right', va='top')

    save_and_show(
        f"tx_classif_results_{corpus_name}_{extractor_name}",
        sub_dir="tx_classif_results"
    )


def display_and_show_perfs(
    verbosity: int,
    labels: PipelineLabels,
    scores: PipelineScores,
    dims: PipelineDims,
    times: PipelineTimes,
) -> None:
    """Display performance metrics and visualization.

    Parameters
    ----------
    verbosity : int
        The verbosity level. The higher the value, the more information will be displayed.
    labels : PipelineLabels
        The labels of the pipeline.
    scores : PipelineScores
        The scores of the pipeline.
    dims : PipelineDims
        The dimensions of the pipeline.
    times : PipelineTimes
        The execution times of the pipeline.

    Returns
    -------
    None

    """
    if verbosity > 1:
        print(f"{bold('n_docs')}: {dims.n_docs}")
        print(f"{bold('n_sents')}: {dims.n_sents}")
        print(f"{bold('n_features')}: {dims.n_feats}")
        print(f"{bold('n_red_feats')}: {dims.n_rd_feats}")
        print(f"{bold('ARI (pred by sent)')}: {scores.ari:.4f}")
        print(f"{bold('ARI (pred by doc)')}: {scores.comb_ari:.4f}")
        print(f"{bold('R2 (pred by sent)')}: {scores.r2:.4f}")
        print(f"{bold('R2 (pred by doc)')}: {scores.comb_r2:.4f}")

        if verbosity > 2:
            for k, v in times._asdict().items():
                title = bold(k.title().replace("_", " "))
                print(f"{title}: {pretty_timedelta_str(v)}")
        else:
            print(
                f"{bold('Total time')}: "
                f"{pretty_timedelta_str(times.total_time)}"
            )

        cla_labels = labels.cla_labels
        clu_labels = labels.clu_labels
        show_confusion_matrix(cla_labels, clu_labels)
        display_classification_report(cla_labels, clu_labels)
        if verbosity > 2:
            show_multilabel_confusion_matrixes(cla_labels, clu_labels)


def _tx_ml_pipeline(
    model_name: str,
    ex_class: Type[BaseEstimator],
    rd_class: Type[BaseEstimator],
    cl_class: Type[BaseEstimator],
    cp_params: Dict[str, Any],
    ex_params: Dict[str, Any],
    rd_params: Dict[str, Any],
    cl_params: Dict[str, Any],
    corpus: pd.Series,
    y: pd.Series,
    corpus_name: str,
    verbosity: int = 0,
    cp_params_update: Optional[Dict[str, Any]] = None,
    ex_params_update: Optional[Dict[str, Any]] = None,
    rd_params_update: Optional[Dict[str, Any]] = None,
    cl_params_update: Optional[Dict[str, Any]] = None,
) -> Tuple:
    r"""Train a composite model consisting of a feature extraction model, 
    a dimensionality reduction model, and a classification model on a corpus
    and return classification results and performance metrics.

    Parameters
    ----------
    model_name : str
        A string representing the name of the model.
    ex_class : Type[BaseEstimator]
        A class object representing the feature extraction model to use.
    rd_class : Type[BaseEstimator]
        A class object representing the dimensionality reduction model to use.
    cl_class : Type[BaseEstimator]
        A class object representing the classification model to use.
    cp_params : Dict[str, Any]
        Optional dictionary containing any parameters for the composite model,
        by default None.
    ex_params : Dict[str, Any]
        Dictionary containing any parameters for the feature extraction model,
        by default None.
    rd_params : Dict[str, Any]
        Dictionary containing any parameters for the dimensionality reduction
        model, by default None.
    cl_params : Dict[str, Any]
        Dictionary containing any parameters for the classification model,
        by default None.
    corpus : pd.Series
        A Pandas series containing the corpus to be classified.
    y : pd.Series
        A Pandas series containing the true labels of the corpus.
    corpus_name : str
        A string representing the name of the corpus.
    verbosity : int, optional
        An integer specifying the level of verbosity (0, 1, 2), by default 0.
    cp_params_update : Optional[Dict[str, Any]], optional
        Optional dictionary containing any updated parameters for the composite
        model, by default None.
    ex_params_update : Optional[Dict[str, Any]], optional
        Optional dictionary containing any updated parameters for the feature
        extraction model, by default None.
    rd_params_update : Optional[Dict[str, Any]], optional
        Optional dictionary containing any updated parameters for the
        dimensionality reduction model, by default None.
    cl_params_update : Optional[Dict[str, Any]], optional
        Optional dictionary containing any updated parameters for the
        classification model, by default None.

    Returns
    -------
    Tuple[pd.Series, Dict[str, float], Tuple[int, int, int, int], ..
    .. Dict[str, float], Tuple[BaseEstimator, BaseEstimator, BaseEstimator]]
        A tuple containing the predicted labels, performance metrics,
        dimensions, elapsed times, and the composite model.
    """
    tt_t = -timeit.default_timer()
    n_sents = corpus.shape[0]
    n_docs = len(set(corpus.index))
    cp_params['name'] = corpus_name
    v = verbosity

    # TODO : intégrer le preprocessing dans le pipeline

    tl(v, f"{model_name} - {ex_class.__name__} ({corpus_name})")
    
    stl(v, "Feature extraction")

    ex, ex_t = create_model(v, ex_class, ex_params, ex_params_update)

    sstl(v, "Extract features")
    feats = ex.fit_transform(corpus)
    ex_t += timeit.default_timer()
    n_feats = feats.shape[1]
    display_features_index(v, ex)
    display_time(v, "extract", ex_t)
   
    stl(v, "Dimensionality reduction")

    rd, rd_t = create_model(v, rd_class, rd_params, rd_params_update)

    sstl(v, "Reduce dimensionality")
    rd_feats = rd.fit_transform(feats)
    rd_t += timeit.default_timer()
    n_rd_feats = rd_feats.shape[1]
    # display_features_index(v, rd)
    display_time(v, "reduce", rd_t)


    stl(v, "Classes prediction")

    cl, cl_t = create_model(v, cl_class, cl_params, cl_params_update)

    sstl(v, "Predict classes")
    cl.fit(rd_feats)
    y_pred = cl.labels_
    cl_t += timeit.default_timer()
    display_time(v, "clusterize", rd_t)

    stl(v, "Predictions combination and labels alignment")

    mapping = match_class(y_pred, y)
    y_pred = np.array([mapping[clu] for clu in y_pred])
    comb_y, comb_y_pred = combine_predictions(y, y_pred)

    stl(v, "Compute scores")

    sh_t = -timeit.default_timer()
    ari = metrics.adjusted_rand_score(y, y_pred)
    comb_ari = metrics.adjusted_rand_score(comb_y, comb_y_pred)
    # Not relevant with classification
    # r2 = metrics.r2_score(y, y_pred)
    # comb_r2 = metrics.r2_score(comb_y, comb_y_pred)

    stl(v, "Show classification results and performances")

    params = pack_pipeline_params(
        ex_class, rd_class, cl_class,
        cp_params, ex_params, rd_params, cl_params,  
    )
    models = pack_pipeline_models(ex, rd, cl)
    scores = pack_pipeline_scores(ari, comb_ari)  # Not relevant with classification, r2, comb_r2)
    dims = pack_pipeline_dims(n_docs, n_sents, n_feats, n_rd_feats)
    labels = pack_pipeline_labels(y, y_pred, comb_y, comb_y_pred)

    show_classif_results(v, rd_feats, labels, scores, params)

    sh_t += timeit.default_timer()
    tt_t += timeit.default_timer()
    times = pack_pipeline_times(ex_t, rd_t, cl_t, sh_t, tt_t)

    display_and_show_perfs(v, labels, scores, dims, times)

    return labels, scores, dims, times, models


def tx_ml_bow_count(
    sents: pd.Series,
    cla_labels: pd.Series,
    corpus_name: str,
    verbosity: int = 0,
    corpus_params: Optional[Dict[str, Any]] = None,
    vectorizer_params: Optional[Dict[str, Any]] = None,
    kmeans_params: Optional[Dict[str, Any]] = None,
    tsne_params: Optional[Dict[str, Any]] = None
) -> Tuple[pd.Series, float, float, int]:
    return _tx_ml_pipeline(
        "Bag of Words",
        CountVectorizer, TSNE, KMeans,
        cp_params={},
        ex_params={
            'stop_words': None,  # already done by preprocessing (NLTK english stopwords)
            'strip_accents': None,  # already done by preprocessing (casefolding)
            'lowercase': False,  # idem
            'max_df': 0.95,  # corpus specific stopwords : too frequent to be relevant
            'min_df': 1,  # a term that appear only once is not relevant
            'ngram_range': (1, 1),  # seems relevant with our long sentences corpus, but it is not
        },
        rd_params={
            'n_components': 2,  # Dimension of the embedded space : fixed
            'perplexity': 30.0,  # [5, 50] Larger datasets usually require a larger perplexity.
            'early_exaggeration': 12.0,  # Not very critical
            'learning_rate': 'auto',  # [10.0, 1000.0],  max(N / early_exaggeration / 4, 50)
            'n_iter': 1000,  # Maximum number of iterations for the optimization (> 250)
            'init': 'random',  # Otherwise 'pca', but not with a sparse input matrix like in this project
            'method': 'barnes_hut',  # Otherwise 'exact' but on small sample (O(N^2) against O(NlogN))
            'random_state': 42,
        },
        cl_params={
            'n_clusters': len(set(cla_labels)),  # Number of clusters
            'n_init': 100,  # Number of time the k-means algorithm will be run with different centroid seeds
            'random_state': 42,
        },
        corpus=sents,
        y=cla_labels,
        corpus_name=corpus_name,
        verbosity=verbosity,
        cp_params_update=corpus_params,
        ex_params_update=vectorizer_params,
        rd_params_update=tsne_params,
        cl_params_update=kmeans_params,
    )


"""def deprecated_tx_ml_bow_count(
    sents: pd.Series,
    cla_labels: pd.Series,
    corpus_name: str,
    verbosity: int = 0,
    corpus_params: Optional[Dict[str, Any]] = None,
    vectorizer_params: Optional[Dict[str, Any]] = None,
    kmeans_params: Optional[Dict[str, Any]] = None,
    tsne_params: Optional[Dict[str, Any]] = None
) -> Tuple[pd.Series, float, float, int]:"""
r"""Apply Bag of Words feature extraction using the CountVectorizer to the
preprocessed sentences.

DEPRECATED

Parameters
----------
sents : pd.Series
    The preprocessed sentences as a series of strings.
cla_labels : pd.Series
    The corresponding class labels.
corpus_name : str
    The corpus name.
verbosity : int, optional
    Verbosity level (default=0).
vectorizer_params : dict, optional
    Additional parameters for CountVectorizer (default=None).
tsne_params : dict, optional
    Additional parameters for tSNE (default=None).

Returns
-------
Tuple[pd.Series, float, float, int]
    The cluster labels, the adjusted Rand index, the processing time and
    the number of features.


Notes
-----
This function applies Bag of Words feature extraction using the
CountVectorizer to the preprocessed sentences. It then applies
t-Distributed Stochastic Neighbor Embedding (tSNE) to reduce the
dimensionality of the feature vectors to 2, and performs k-means clustering
on the reduced data. The combined predictions from clustering at the
sentence level and clustering at the document level are then used to
calculate the adjusted Rand index.
"""
"""    start_t = time.time()
    n_sents = sents.shape[0]
    n_docs = len(set(sents.index))

    if verbosity > 0:
        print_title(f"Bag of words - CountVectorizer ({corpus_name})")

    if verbosity > 1:
        print_subtitle("Create the CountVectorizer object")
    
    _vectorizer_params = {
        'stop_words': None,  # already done by preprocessing (NLTK english stopwords)
        'strip_accents': None,  # already done by preprocessing (casefolding)
        'lowercase': False,  # idem
        'max_df': 0.95,  # corpus specific stopwords : too frequent to be relevant
        'min_df': 1,  # a term that appear only once is not relevant
        'ngram_range': (1, 1),  # seems relevant with our long sentences corpus, but it is not
    }
    _vectorizer_params.update(vectorizer_params or {})
    vectorizer = CountVectorizer(**_vectorizer_params)

    if verbosity > 2:
        display(vectorizer)
        print("params:")
        display(vectorizer.get_params())

    if verbosity > 1:
        print_subtitle("Feature extraction")

    features = vectorizer.fit_transform(sents)
    n_features = features.shape[1]

    if verbosity > 2:
        print("n_features:", features.shape[1])
        display(vectorizer.get_feature_names_out())

    if verbosity > 1:
        print_subtitle("tSNE and ARI")
    
    X_tsne, clu_labels, ari, _, _ = tsne_kmeans_ari(
        features,
        cla_labels,
        verbosity=0,
        tsne_params=tsne_params
    )

    if verbosity > 1:
        print_subtitle("Combine predictions")

    comb_cla_labels, comb_clu_labels = combine_predictions(cla_labels, clu_labels)
    comb_ari = metrics.adjusted_rand_score(comb_cla_labels, comb_clu_labels)

    r2 = metrics.r2_score(cla_labels, clu_labels)
    comb_r2 = metrics.r2_score(comb_cla_labels, comb_clu_labels)

    if verbosity > 0:
        print_subtitle("Plot")
        cla_names = list(get_class_label_name_map().values())
        results = {
            'n_features': n_features,
            'comb_ari': round(comb_ari, 2),
            'ari': round(ari, 2),
            'comb_r2': round(comb_r2, 2),
            'r2': round(r2, 2),
        }
        params = {
            'corpus': corpus_params,
            'extractor': vectorizer_params,
            'kmeans': kmeans_params,
            'tsne': tsne_params,
        }
        vectorizer_params['name'] = 'CountVectorizer'
        show_tsne(cla_labels, cla_names, X_tsne, clu_labels, results, params)

    dt = time.time() - start_t

    ""display_and_show_perfs(
        verbosity,
        n_docs, n_sents, n_features,
        comb_cla_labels, comb_clu_labels,
        start_t,
        ari, comb_ari, r2, comb_r2
    )""

    return comb_clu_labels, comb_ari, dt, n_features
"""


def tx_ml_bow_tfidf(
    sents: pd.Series,
    cla_labels: pd.Series,
    corpus_name: str,
    verbosity: int = 0,
    corpus_params: Optional[Dict[str, Any]] = None,
    vectorizer_params: Optional[Dict[str, Any]] = None,
    kmeans_params: Optional[Dict[str, Any]] = None,
    tsne_params: Optional[Dict[str, Any]] = None
) -> Tuple[pd.Series, float, float, int]:
    r"""Apply Bag of Words feature extraction using the TfidfVectorizer to the
    preprocessed corpus."""
    return _tx_ml_pipeline(
        "Bag of Words",
        TfidfVectorizer, TSNE, KMeans,
        cp_params={},
        ex_params={
            'stop_words': None,  # already done by preprocessing (NLTK english stopwords)
            'strip_accents': None,  # already done by preprocessing (casefolding)
            'lowercase': False,  # idem
            'max_df': 0.95,  # corpus specific stopwords : too frequent to be relevant
            'min_df': 1,  # a term that appear only once is not relevant
            'ngram_range': (1, 1),  # seems relevant with our long sentences corpus, but it is not
            # Preceding parameters are in common with the CountVectorizer
            'norm': 'l2',  # Otherwise 'l1'
            'sublinear_tf': False,  # If True tf becomes 1 + log(tf)
        },
        rd_params={
            'n_components': 2,  # Dimension of the embedded space : fixed
            'perplexity': 30.0,  # [5, 50] Larger datasets usually require a larger perplexity.
            'early_exaggeration': 12.0,  # Not very critical
            'learning_rate': 'auto',  # [10.0, 1000.0],  max(N / early_exaggeration / 4, 50)
            'n_iter': 1000,  # Maximum number of iterations for the optimization (> 250)
            'init': 'random',  # Otherwise 'pca', but not with a sparse input matrix like in this project
            'method': 'barnes_hut',  # Otherwise 'exact' but on small sample (O(N^2) against O(NlogN))
            'random_state': 42,
        },
        cl_params={
            'n_clusters': len(set(cla_labels)),  # Number of clusters
            'n_init': 100,  # Number of time the k-means algorithm will be run with different centroid seeds
            'random_state': 42,
        },
        corpus=sents,
        y=cla_labels,
        corpus_name=corpus_name,
        verbosity=verbosity,
        cp_params_update=corpus_params,
        ex_params_update=vectorizer_params,
        rd_params_update=tsne_params,
        cl_params_update=kmeans_params,
    )


class KerasW2VModel:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.w2v_model = None

    def get_params(self):
        if self.w2v_model:
            return get_w2v_hyperparams(self.w2v_model)
        else:
            return self.params
    

    def get_feature_names_out(self):
        if self.w2v_model is None:
            return None
        else:
            return self.w2v_model.wv.index_to_key


    def fit_transform(self, corpus):
        # Word tokenize the corpus
        tok_corpus = gesim_tokenize(corpus)
        # Create and train the Word2Vec model using half of cpus
        self.params["workers"] = max(1, os.cpu_count() // 2)
        self.w2v_model = create_w2v_model(tok_corpus, **self.params)
        # Numerize ans pad tok_corpus
        keras_tokenizer, padded_num_corpus = keras_num_and_pad(tok_corpus)
        # Create the word embedding matrix
        embed_mx = get_embedding_matrix(self.w2v_model, keras_tokenizer)
        # Builds the Keras model
        keras_w2v_model = get_keras_w2v_embedding_model(
            padded_num_corpus,
            self.w2v_model,
            keras_tokenizer,
            embed_mx
        )
        # Return the features
        return keras_w2v_model.predict(padded_num_corpus)


class BertVectorizer:

    def __init__(
        self,
        model_type: str,
        output_hidden_states: bool = False,
        hub: str = "huggingface",
        max_length: int = 64,
        batch_size: int = None,   # `None` means no batch
        include_cls_token: bool = False,  # If True, get the [CLS] token
    ):
        self.model_type = model_type
        self.hub = hub
        self.max_length = max_length
        self.batch_size = batch_size
        self.include_cls_token = include_cls_token

        if hub == "huggingface":
            self.bert_model = TFAutoModel.from_pretrained(
                model_type, output_hidden_states=output_hidden_states
            )
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
            self.output_key = "last_hidden_state"
            if include_cls_token:
                self.pooler_key = "pooler_output"
            else:
                self.pooler_key = None

        elif hub == "tfhub":
            self.bert_model = hub.KerasLayer(
                f"https://tfhub.dev/tensorflow/{model_type}/3",
                trainable=True
            )
            self.bert_tokenizer = FullTokenizer(
                vocab_file=f"gs://tfhub-modules/tensorflow/{model_type}/3/assets/30k-clean.vocab",
                do_lower_case=True
            )
            self.output_key = "sequence_output"
            if include_cls_token:
                self.pooler_key = "pooled_output"
            else:
                self.pooler_key = None

        else:
            raise ValueError(f"Invalid hub value: {hub}")

        self.n_features = self.bert_model.output_shape[-1]

    def get_params(self):
        return {
            "model_type": self.model_type,
            "max_length": self.max_length,
            "batch_size": self.batch_size
        }

    def get_feature_names_out(self):
        return [f"feat_{i}" for i in range(self.n_features)]

    def fit_transform(self, corpus: pd.Series) -> np.ndarray:
        # Encode the corpus with the BERT tokenizer
        encoded_corpus = encode_sentences_with_bert(
            corpus, self.bert_tokenizer, self.max_length
        )

        # Get the BERT embeddings for the BERT encoded corpus
        embeddings = None
        if self.batch_size is None:  # No batch mode
            if self.hub == "huggingface":
                if self.include_cls_token:
                    outputs = self.bert_model.predict(encoded_corpus)
                    embeddings = outputs[self.pooler_key].numpy()
                else:
                    outputs = self.bert_model(encoded_corpus)
                    embeddings = outputs[self.output_key].numpy()

            elif self.hub == "tfhub":
                tf_encoded_corpus = {
                    "input_word_ids": encoded_corpus[0],
                    "input_type_ids": encoded_corpus[1],
                    "input_mask": encoded_corpus[2],
                }
                outputs = self.bert_model(tf_encoded_corpus)
                embeddings = outputs[self.output_key].numpy()

        else:  # batch mode
            n_samples = corpus.shape[0]
            embeddings = np.empty((n_samples, self.n_features))
            for i in range(0, n_samples, self.batch_size):
                batch_slice = slice(i, i+self.batch_size)
                batch = tuple(
                    encoded_corpus[i][batch_slice]
                    for i in range(3)
                )
                outputs = self.bert_model(batch)
                # embeddings[batch_slice] = batch_embeddings[1]
                if self.include_cls_token:
                    embeddings[batch_slice] = outputs[self.pooler_key]
                else:
                    embeddings[batch_slice] = outputs[self.output_key]

        # Compute and return the mean of the word embeddings for each sentence
        return np.mean(embeddings, axis=1)



def tx_ml_word2vec(
    sents: pd.Series,
    cla_labels: pd.Series,
    corpus_name: str,
    verbosity: int = 0,
    corpus_params: Optional[Dict[str, Any]] = None,
    word2vec_params: Optional[Dict[str, Any]] = None,
    kmeans_params: Optional[Dict[str, Any]] = None,
    tsne_params: Optional[Dict[str, Any]] = None
) -> Tuple[pd.Series, float, float, int]:
    r"""Apply Word2Vec embedding to the preprocessed corpus."""
    return _tx_ml_pipeline(
        "Word2Vec",
        KerasW2VModel, TSNE, KMeans,
        cp_params={},
        ex_params={
        # Algo choice
            'cbow': True,  # Training algorithm: True for CBOW (Continuous Bag Of Words).
                           # Otherwise Skip-Gram
        # Word space dim
            'vector_size': 100,   # Dimensionality of the word space
                                  # Recommended : large corpus [50, 300], small corpus [100, 1000]
        # Freq thresholding
            'min_df': 5,  # Ignores all words with total frequency lower than this.
                          # Recommended value : large corpus [1, 5], small corpus [5, 20]
            'hf_thres': 1e-3,  # [0.0, 1.0] The threshold for configuring which higher-frequency words
                               # are randomly downsampled (probability of sampling a word with df > df_min)
                               # Recommanded value : [1e-5, 1e-3] => np.logspace(-5, -3, num=10)
        # Context window size
            'window': 5,  # [2, 15] Context window : maximum distance between the current
                          # and predicted word within a sentence.
                          # Recommended value is 10 for skip-gram and 5 for CBOW
                          # [2, 7] for CBOW, [8, 12] for Skip-Gram
        # Option for CBOW mean against sum
            'cbow_mean': True,  # If False, use the sum of the context word vectors.
                                # If True, use the mean (generaly a best choice).
                                # Only applies when cbow is used.
        # Activate the hierarchical softmax (=> no negative sampling)
            'hs': False,  # If True, hierarchical softmax will be used for model training (no negative sampling).
                          # If False, and `negative` is non-zero, negative sampling will be used.
        # Negative sampling
            'negative': 5,  # [0, 20] If > 0, negative sampling will be used.
                            # The int for negative specifies how many "noise words"
                            # should be drawn (usually between 5-20).
                            # Default is 5. If set to 0, no negative samping is used.
                            # small corpus [0, 5], large corpus [5, 20]
            'ns_exponent': 0.75,  # [0.0, 1.0] The exponent used to shape the negative sampling distribution.
                                  # A value of 1.0 samples exactly in proportion to the frequencies,
                                  # 0.0 samples all words equally, while a value of less than 1.0
                                  # skews towards low-frequency words.
        # Learning rate
            'alpha': 0.03,        # The initial learning rate.
            'min_alpha': 0.0007,  # Learning rate will linearly drop to `min_alpha` as training progresses.
                                  # small corpus [5e-3, 1e-2] x [5e-4, 1e-3]
                                  # middle corpus [1e-2, 5e-2] x [5e-4, 1e-3]
                                  # large corpus [5e-2, 1e-1] x [5e-4, 1e-3]
            'n_iter': 5,  # Number of iterations (epochs) over the corpus.
                          # small [5, 10], middle [10, 20], large [20, 100]
        # Vocabulary sorting (or not)
            'sorted_vocab': True,  # If True, sort the vocabulary by descending frequency before assigning word indices.
                                   # If False can be semanticaly performant but costly.
                                   # If False, max_vocab_size must be specified (not None)
            'max_vocab_size': None,  # Limit RAM during vocabulary building; if there are more unique words than this,
                                     # then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
                                     # Set to None for no limit.
        # More
            'random_state': 42,  # Seed for the random number generator.
            'compute_loss': False,  # If True, compute and report the training loss.
        },
        rd_params={
            'n_components': 2,  # Dimension of the embedded space : fixed
            'perplexity': 30.0,  # [5, 50] Larger datasets usually require a larger perplexity.
            'early_exaggeration': 12.0,  # Not very critical
            'learning_rate': 'auto',  # [10.0, 1000.0],  max(N / early_exaggeration / 4, 50)
            'n_iter': 1000,  # Maximum number of iterations for the optimization (> 250)
            'init': 'random',  # Otherwise 'pca', but not with a sparse input matrix like in this project
            'method': 'barnes_hut',  # Otherwise 'exact' but on small sample (O(N^2) against O(NlogN))
            'random_state': 42,
        },
        cl_params={
            'n_clusters': len(set(cla_labels)),  # Number of clusters
            'n_init': 100,  # Number of time the k-means algorithm will be run with different centroid seeds
            'random_state': 42,
        },
        corpus=sents,
        y=cla_labels,
        corpus_name=corpus_name,
        verbosity=verbosity,
        cp_params_update=corpus_params,
        ex_params_update=word2vec_params,
        rd_params_update=tsne_params,
        cl_params_update=kmeans_params,
    )


def _check_gpu() -> None:
    r"""Check the available GPUs and display some information about Tensorflow version"""
    print("Tensorflow version:", tf.__version__)
    n_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available:", n_gpu)
    print("Built with cuda:", tf.test.is_built_with_cuda())


def tx_ml_bert(
    sents: pd.Series,
    cla_labels: List[int],
    name: str
) -> None:
    r"""Extract sentence embeddings using BERT HuggingFace and visualize the
    data with t-SNE.

    Parameters:
    -----------
    sents : pd.Series
        The preprocessed sentences as a series of strings to be transformed
        into embeddings.
    cla_labels: list
        A list of integers representing the labels of each sentence.
    name: str
        A string representing the name of the model being used.

    Returns:
    --------
    None
    """
    start_t = time.time()
    n_sents = sents.shape[0]
    n_docs = len(set(sents.index))

    print_title(f"Embedding - BERT HuggingFace ({name})")

    _check_gpu()

    # BERT HuggingFace
    model_type = 'bert-base-uncased'
    model = TFAutoModel.from_pretrained(model_type)

    print_subtitle("Feature extraction")
    max_length = 64
    batch_size = 10
    features, _ = extract_bert_sentence_embeddings(
        model, model_type,
        sents, 
        max_length, batch_size, mode='HF'
    )
    n_features = features.shape[1]

    print_subtitle("tSNE and ARI")
    X_tsne, clu_labels, ari = tsne_kmeans_ari(features, cla_labels)

    print_subtitle("Plot")
    cla_names = list(get_class_label_name_map().values())
    show_tsne(
        cla_labels, cla_names, X_tsne, clu_labels, ari,
        f"BERT embedded {name} t-SNE clustering"
    )


def tx_ml_use(
    sents: pd.Series,
    cla_labels: List[int],
    name: str
) -> None:
    r"""Extract sentence embeddings using USE and visualize the data with
    t-SNE.

    Parameters:
    -----------
    sents : pd.Series
        The preprocessed sentences as a series of strings to be transformed
        into embeddings.
    cla_labels: list
        A list of integers representing the labels of each sentence.
    name: str
        A string representing the name of the model being used.

    Returns:
    --------
    None
    """
    start_t = time.time()
    n_sents = sents.shape[0]
    n_docs = len(set(sents.index))

    print_title(f"Embedding - USE  ({name})")

    _check_gpu()

    print_subtitle("Feature extraction")
    features = extract_use_sentence_embeddings(sents)
    n_features = features.shape[1]

    print_subtitle("tSNE and ARI")
    X_tsne, clu_labels, ari = tsne_kmeans_ari(features, cla_labels)

    if verbosity > 1:
        print_subtitle("Combine predictions")

    comb_cla_labels, comb_clu_labels = combine_predictions(cla_labels, clu_labels)
    comb_ari = metrics.adjusted_rand_score(comb_cla_labels, comb_clu_labels)

    print_subtitle("Plot")
    cla_names = list(get_class_label_name_map().values())
    show_tsne(
        cla_labels, cla_names, X_tsne, clu_labels, ari,
        f"USE embedded {name} t-SNE clustering"
    )

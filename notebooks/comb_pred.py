from typing import *

import numpy as np
import pandas as pd

from tx_ml import match_class


def combine(
    preds: List[int],
    probas: bool = False
) -> Union[Tuple[List[int], List[float]], int]:
    r"""Combine the predictions of different models.

    Parameters
    ----------
    preds : List[int]
        A list of integer class labels.
    probas : bool, optional (default=False)
        If True, return the normalized class probabilities for each unique
        class label. Otherwise, return the most frequently occurring class
        label.

    Returns
    -------
    Union[Tuple[List[int], List[float]], int]
        If `probas=True`, return a tuple containing a list of unique class
        labels and their normalized probabilities. If `probas=False`, return
        the most frequently occurring class label.
    """
    cls, cl_counts = np.unique(preds, return_counts=True)
    n = np.sum(cl_counts)
    return (cls, cl_counts / n) if probas else cls[0]


def lst_inflate_probas(
    cl_probas: Tuple[List[int], List[float]],
    n_cl: int
) -> List[float]:
    r"""Inflate the probabilities for some classes.

    Parameters
    ----------
    cl_probas : Tuple[List[int], List[float]]
        A tuple containing the unique class labels and their corresponding
        probabilities.
    n_cl : int
        The total number of classes.

    Returns
    -------
    List[float]
        A list containing the inflated probabilities for all classes, where
        the probabilities for the classes not present in `cl_probas` are set to
        0.
    """
    labels, probas = cl_probas
    return [
        0.0 if cl not in labels
        else probas[labels.index(cl)]
        for cl in range(n_cl)
    ]


def np_inflate_probas(
    cl_probas: Tuple[List[int], List[float]],
    n_cl: int
) -> np.ndarray:
    """Inflate the probabilities for some classes.

    Parameters
    ----------
    cl_probas : Tuple[List[int], List[float]]
        A tuple containing the unique class labels and their corresponding
        probabilities.
    n_cl : int
        The total number of classes.

    Returns
    -------
    np.ndarray
        An array containing the inflated probabilities for all classes, where
        the probabilities for the classes not present in `cl_probas` are set to
        0.
    """
    labels, probas = cl_probas
    inflated_probas = np.zeros(n_cl)
    mask = np.isin(np.arange(n_cl), labels)
    inflated_probas[mask] = probas
    return inflated_probas


def combine_predictions(
    cla_labels: pd.Series,
    clu_labels: np.ndarray,
    align: bool = True,
    probas: bool = False
) -> Union[Tuple[pd.Series, pd.DataFrame], Tuple[pd.Series, pd.Series]]:
    r"""Combine the class labels and/or probabilities of two sets of
    predictions.

    Parameters
    ----------
    cla_labels : pd.Series
        A series containing the ground truth class labels.
    clu_labels : np.ndarray
        An array containing the predicted class labels.
    align : bool, optional (default=True)
        If True, align the predicted class labels with the ground truth labels
        using the Hungarian algorithm.
    probas : bool, optional (default=False)
        If True, return the normalized class probabilities for each unique
        class label. Otherwise, return the most frequently occurring class
        label.

    Returns
    -------
    Tuple[pd.Series, pd.Series] or Tuple[pd.Series, pd.DataFrame]
        If probas is False, returns a tuple containing a Series of combined
        true class labels and a Series of combined predicted class labels.
        If probas is True, returns a tuple containing a Series of combined
        true class labels and a DataFrame of combined predicted class
        probabilities, where each row represents a unique id from cla_labels.
    """
    if align:
        mapping = match_class(clu_labels, cla_labels)
        clu_labels = np.array([mapping[clu] for clu in clu_labels])

    cla_labels.rename('cla', inplace=True)
    data = cla_labels.reset_index()
    data["clu"] = clu_labels
    
    gpby = data.groupby(by='id').agg({
        'cla': combine,
        'clu': lambda x: combine(x, probas)
    })

    if probas:
        n_cl = len(set(cla_labels))
        clu_probas = gpby.clu.apply(lambda x: np_inflate_probas(x, n_cl))
        return gpby.cla, clu_probas.apply(pd.Series)
    else:
        return gpby.cla, gpby.clu


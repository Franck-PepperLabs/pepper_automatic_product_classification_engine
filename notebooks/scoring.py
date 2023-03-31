from typing import *
from pepper_utils import print_subtitle
from flipkart_utils import get_class_label_name_map
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def show_confusion_matrix(
    cla_labels: List[str],
    clu_labels: List[str]
) -> None:
    r"""Show confusion matrix with predicted and true class labels.

    Parameters
    ----------
    cla_labels : list
        List of true class labels.
    clu_labels : list
        List of predicted class labels.

    Returns
    -------
    None

    Note
    ----
    There is a SKL built in alternative :
    >>> from sklearn.metrics import ConfusionMatrixDisplay
    >>> fig, ax = plt.subplots(figsize=(10, 5))
    >>> ConfusionMatrixDisplay.from_predictions(
    >>>     cla_labels, aligned_clu_labels, ax=ax
    >>> )
    >>> ax.xaxis.set_ticklabels(cla_names)
    >>> ax.yaxis.set_ticklabels(cla_names)
    >>> _ = ax.set_title(f"Confusion Matrix")
    """
    print_subtitle("Confusion matrix")
    conf_mx = metrics.confusion_matrix(cla_labels, clu_labels)
    cla_names = [
        cln[:13] + ("..." if len(cln) > 16 else cln[13:16])
        for cln in list(get_class_label_name_map().values())
    ]
    conf_data = pd.DataFrame(conf_mx, index=cla_names, columns=cla_names)

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_data, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.title("Confusion matrix", fontsize=15, pad=15, weight="bold")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.show()


def display_classification_report(cla_labels: list, clu_labels: list) -> None:
    r"""Display classification report with metrics for each class.

    Parameters
    ----------
    cla_labels : list
        List of true class labels.
    clu_labels : list
        List of predicted class labels.

    Returns
    -------
    None
    """
    print_subtitle("Classification report")
    print(metrics.classification_report(cla_labels, clu_labels))


def show_multilabel_confusion_matrixes(
    cla_labels: list,
    clu_labels: list
) -> None:
    r"""Show multiple confusion matrices for multilabel classification.

    Parameters
    ----------
    cla_labels : list
        List of true class labels.
    clu_labels : list
        List of predicted class labels.

    Returns
    -------
    None
    """
    def _plot_conf_mx(cfmx, cla_name, ax):
        sns.heatmap(cfmx, annot=True, fmt="d", linewidths=.5, ax=ax)
        ax.set_title(f"{cla_name} ConfMx")

    print_subtitle("multilabel confusion matrixes")
    labels = list(set(cla_labels))
    cla_names = list(get_class_label_name_map().values())
    conf_mx = metrics.multilabel_confusion_matrix(
        cla_labels, clu_labels, labels=labels
    )

    fig = plt.figure(figsize=(15, 7))
    for i in range(7):
        _plot_conf_mx(conf_mx[i], cla_names[i], ax=fig.add_subplot(240 + i + 1))
    plt.suptitle("Multilabel Confusion Matrixes", fontsize=15, weight="bold")
    plt.show()

import os
from typing import *
from itertools import zip_longest
from IPython.display import display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def discrete_stats(data, name=None):
    """[count, unique_count, na_count, filling_rate, variety_rate]
    as [n, n_u, n_na, fr, vr] for each var in data
    """
    n = data.count()
    n_u = data.nunique()
    n_na = data.isna().sum()
    stats = pd.DataFrame({
        'n': n,
        'n_u': n_u,
        'n_na': n_na,
        'Filling rate': n / (n + n_na),
        'Shannon entropy': n_u / n,
        'dtypes': data.dtypes
    }, index=data.columns)
    if name is not None:
        stats.index.names = [name]

    return stats


def plot_discrete_stats(stats, precision=.1):
    table_name = stats.index.name
    filling_rate = stats[['Filling rate', ]].copy()
    na_rate = 1 - filling_rate['Filling rate']
    filling_rate.insert(0, 'NA_', na_rate)
    filling_rate = filling_rate * 100
    filling_rate.columns = ['NA', 'Filled']

    shannon_entropy = stats['Shannon entropy']
    shannon_entropy = np.maximum(shannon_entropy * 100, precision)

    # Create stacked bar chart
    ax1 = filling_rate.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'])
    legend1 = ax1.legend(['NA', 'Filled'], loc='upper left', bbox_to_anchor=(1, 1))
    plt.gca().add_artist(legend1)

    # Add scatter plot for Shannon entropy
    ax2 = plt.scatter(
        np.arange(len(filling_rate)),  # x-coordinates
        shannon_entropy,               # y-coordinates
        s=200,                         # size of the points
        color='black'
    )
    plt.legend([ax2], ['Shannon entropy'], loc='upper left', bbox_to_anchor=(1, .8))

    plt.yscale('log')
    plt.ylim(precision, 100)

    # Axis titles
    plt.ylabel('Filling rate & Shannon entropy')
    plt.xlabel('')

    # Rotate x-axis labels
    plt.xticks(rotation=30, ha='right')

    # Add overall title
    plt.title(f'Discrete statistics of `{table_name}` table', fontsize=16)

    plt.savefig(
        f'../img/Filling rate & Shannon entropy of `{table_name}`.png',
        facecolor='white',
        bbox_inches='tight',
        dpi=300   # x 2
    )

    plt.show()


def show_discrete_stats(data, name=None):
    stats = discrete_stats(data, name=name)
    display(stats)
    plot_discrete_stats(stats)


""" Generics
"""

def create_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def for_all(
    f: Callable[..., Any],
    args_vect: Optional[List[Optional[Tuple[Any, ...]]]] = None,
    kwargs_vect: Optional[List[Optional[Dict[str, Any]]]] = None,
    const_args: Optional[Tuple[Any, ...]] = None,
    const_kwargs: Optional[Dict[str, Any]] = None
) -> Union[None, Any, List[Any]]:
    """
    Apply a function to a vector of arguments and/or keyword arguments.

    If `args_vect` and `kwargs_vect` are provided, apply `f(*args_vect[i], **kwargs_vect[i])`
    to each corresponding pair `(args_vect[i], kwargs_vect[i])`. If only `args_vect` or
    `kwargs_vect` is provided, `f(*args_vect[i])` or `f(**kwargs_vect[i])` is applied to each
    element.

    If `const_args` or `const_kwargs` are provided, their values are used as additional arguments
    and/or keyword arguments in each function call.

    Args:
        f (callable): The function to apply to each set of arguments.
        args_vect (list of tuple or None): A vector of positional arguments. Each element of the list
            is a tuple of arguments to be passed to the function. If `None`, `kwargs_vect` must be
            provided.
        kwargs_vect (list of dict or None): A vector of keyword arguments. Each element of the list is
            a dictionary of keyword arguments to be passed to the function. If `None`, `args_vect` must
            be provided.
        const_args (tuple or None): Additional positional arguments to pass to the function in each
            call. If `None`, no additional positional arguments are used.
        const_kwargs (dict or None): Additional keyword arguments to pass to the function in each call.
            If `None`, no additional keyword arguments are used.

    Returns:
        If `args_vect` and `kwargs_vect` are `None` and `const_args` and `const_kwargs` are `None`,
        `None` is returned. Otherwise, a list of return values from `f` is returned.

        If `f` returns a tuple, the list of return values is transposed such that the i-th element of
        the j-th tuple returned by `f` is the j-th element of the i-th tuple in the returned list.

    Examples:
        >>> for_all(len, args_vect=[('abc',), ('def', 'ghi'), ()])
        [1, 2, 0]
        >>> for_all(len, kwargs_vect=[{'x': 'abc'}, {'x': 'def', 'y': 'ghi'}, {}])
        [3, 3, 0]
        >>> for_all(lambda x, y: x + y, args_vect=[(1, 2), (3, 4)], const_args=(10,))
        [13, 14]
        >>> for_all(lambda x, y=0: x + y, args_vect=[(1,), (2,)], kwargs_vect=[{'y': 10}, {}])
        [11, 2]
        >>> for_all(lambda x, y: (x, y), args_vect=[(1, 2), (3, 4), (5,)], const_kwargs={'y': 10})
        [(1, 12), (3, 14), (..
"""
    # Handle the case where no arguments are given
    if (
        args_vect is None
        and kwargs_vect is None
        and const_args is None
        and const_kwargs is None
    ):
        return None

    # Ensure that constant arguments are a tuple
    if not (const_args is None or isinstance(const_args, tuple)):
        const_args = (const_args,)

    # Handle the case where only constant arguments are given
    if args_vect is None and kwargs_vect is None:
        if const_args is None:
            return f(**const_kwargs)
        if const_kwargs is None:
            return f(*const_args)

    # Apply the function to all combinations of variable and constant arguments
    def call_f(args, kwargs):
        new_args = ()
        if args is not None:
            if not isinstance(args, tuple):
                args = (args,)
            new_args += args
        if const_args is not None:
            new_args += const_args
        new_kwargs = {}
        if const_kwargs is not None:
            new_kwargs.update(const_kwargs)
        if kwargs is not None:
            new_kwargs.update(kwargs)
        return f(*new_args, **new_kwargs)

    results = None
    if args_vect is None:
        results = [call_f(None, kwargs) for kwargs in kwargs_vect]
    elif kwargs_vect is None:
        results = [call_f(args, None) for args in args_vect]
    else:
        results = [
            call_f(args, kwargs)
            for args, kwargs in zip(args_vect, kwargs_vect)
        ]

    # If the function returns a tuple, we zip the output
    if len(results) > 0 and isinstance(results[0], tuple):
        results = list(zip_longest(*results))

    # Returns nothing if f is clearly a procedure (never returns anything)
    if not all([result is None for result in results]):
        return results

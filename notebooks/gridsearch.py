from typing import *
import itertools
import numpy as np
import pandas as pd
from IPython.display import display


def build_param_range(
    param_type: type,
    min: float,
    max: float,
    n: int,
    log: bool = False
) -> np.ndarray:
    r"""Builds a range of parameter values between `min_val` and `max_val` with
    `num_vals` values.

    Parameters
    ----------
    param_type : type or np.dtype
        The data type of the output array.
    min_val : float
        The minimum value of the range.
    max_val : float
        The maximum value of the range. Note that this value is inclusive, even
        if `param_type` is an `int`.
    num_vals : int
        The number of values in the range.
    log_scale : bool, optional
        If True, the range is logarithmic.

    Returns
    -------
    np.ndarray
        An array of `num_vals` evenly spaced values between `min_val` and
        `max_val`.
    
    Examples
    --------
    >>> print(build_param_range(int, 100, 300, 3))
    array([100, 200, 300])
    >>> print(build_param_range(int, 100, 300, 4))
    array(100, 166, 233, 300])
    >>> print(build_param_range(int, 1, 3, 4, True))
    array([  10,   46,  215, 1000])
    >>> print(build_param_range(float, 2, 9, 5))
    array([2.  , 3.75, 5.5 , 7.25, 9.  ])
    >>> print(build_param_range(float, 1, 3, 4, True))
    array([  10.        ,   46.41588834,  215.443469  , 1000.        ])
    >>> print(build_param_range(float, 1, 1, 1))
    array([1.])
    >>> print(build_param_range(float, 1, 1, 1, True))
    array([10.])
    >>> print(build_param_range(bool, 0, 1, 2))
    array([False,  True])
    """
    if param_type == int and n > 1 and (max-min) % (n-1) == 0 and not log:
        return np.arange(min, max+1, (max-min) // (n-1))
    rg = np.logspace(min, max, n) if log else np.linspace(min, max, n)
    return rg.astype(param_type)


def test_build_param_range_1():
    display(build_param_range(int, 100, 300, 3))
    display(build_param_range(int, 100, 300, 4))
    display(build_param_range(int, 1, 3, 4, True))
    display(build_param_range(float, 2, 9, 5))
    display(build_param_range(float, 1, 3, 4, True))
    display(build_param_range(float, 1, 1, 1))
    display(build_param_range(float, 1, 1, 1, True))
    display(build_param_range(bool, 0, 1, 2))
    domain = (int, 2, 3, 10, True)
    display(build_param_range(*domain))


def test_param_domain_defs():
    param_domain_defs = {
        "cbow": (bool, 1, 1, 1),
        "vector_size": (int, 2, 3, 10, True),
        "min_df": (int, 5, 5, 1)
    }
    return param_domain_defs


def get_param_domains(param_domain_defs: Dict) -> Tuple:
    r"""Given a dictionary of parameter domain definitions, return a tuple of
    parameter domains.

    Parameters:
    -----------
    param_domain_defs : Dict
        A dictionary of parameter domain definitions. Each key represents a
        parameter name, and the corresponding value is a tuple of the form
        (type, start, end, step). The 'type' specifies the type of the
        parameter (e.g., int or float), 'start' is the starting value of the
        domain, 'end' is the ending value of the domain, and 'step' is the
        increment used to generate the domain.

    param_domain_defs : Dict
        A dictionary of parameter domain definitions. Each key represents a
        parameter name, and the corresponding value is a tuple of the form
        (type, start, end, n), where:
        - 'type' specifies the type of the parameter (e.g., int or float)
        - 'start' is the starting value of the domain
        - 'end' is the ending value of the domain
        - 'n' is the number of elements in the domain

        Alternatively, if the domain consists of a list of specific values, the
        value can be specified directly as a list (e.g., ['a', 'b', 'c']) or a
        single element (e.g. 3.0, 1, 'a' or False).

    Returns:
    --------
    Tuple:
        A tuple of parameter domains, where each domain is an iterator of
        values generated using the corresponding domain definition.

    Example:
    --------
    >>> param_domain_defs = {
    >>>     "m": (int, 1, 10, 5),
    >>>     "n": (int, 1, 10, 7),
    >>>     "x": (float, 1, 5, 3),
    >>>     "helen": (float, 1, 4, 3, True),
    >>>     "b": (bool, 0, True, 2),
    >>>     "p": (int, 130, 130, 1),
    >>>     "q": 1, "r": 2.0, "s": False,
    >>>     "t": ['a', 'b', 'c']
    >>> }
    >>> get_param_domains(param_domain_defs)
    (array([ 1,  3,  5,  7, 10]),
     array([ 1,  2,  4,  5,  7,  8, 10]),
     array([1., 3., 5.]),
     array([   10.        ,   316.22776602, 10000.        ]),
     array([False,  True]),
     array([130]),
     array([1]),
     array([2.]),
     array([False]),
     array(['a', 'b', 'c'], dtype='<U1'))
    """
    return tuple(
        build_param_range(*v) if isinstance(v, tuple)
        else np.array(v if isinstance(v, list) else [v])
        for v in param_domain_defs.values()
    )

def test_build_param_range_2():
    param_domain_defs = test_param_domain_defs()

    for k, v in param_domain_defs.items():
        print(f"{k}: {build_param_range(*v)}")
    
    param_domains = get_param_domains(param_domain_defs)
    display(param_domains)


def get_param_names(param_domain_defs: Dict) -> Tuple:
    r"""Get the names of parameters given a dictionary of parameter domain
    definitions.

    Parameters:
    -----------
    param_domain_defs : Dict
        Dictionary mapping parameter names to their corresponding domain
        definitions.

    Returns:
    --------
    Tuple
        The parameter names.
    """
    return tuple(param_domain_defs.keys())


def test_get_params_names():
    param_domain_defs = test_param_domain_defs()
    param_names = get_param_names(param_domain_defs)
    display(param_names)


def get_combined_params_iterator(param_domain_defs: Dict) -> itertools.product:
    r"""Get an iterator over the parameter domain definitions.

    Parameters:
    -----------
    param_domain_defs : Dict
        Dictionary mapping parameter names to their corresponding domain definitions.

    Returns:
    --------
    itertools.product
        An iterator over the Cartesian product of the parameter domain definitions.
    """
    param_domains = get_param_domains(param_domain_defs)
    return itertools.product(*param_domains)


def test_get_combined_params_iterator():
    param_domain_defs = test_param_domain_defs()
    combined_params_iterator = get_combined_params_iterator(param_domain_defs)
    display(list(combined_params_iterator))


def test_params_all():
    param_domain_defs = test_param_domain_defs()
    param_names = get_param_names(param_domain_defs)
    param_domains = get_param_domains(param_domain_defs)
    for param_name, param_domain in zip(param_names, param_domains):
        print(f"'{param_name}' range: {param_domain}")
    combined_params_iterator = get_combined_params_iterator(param_domain_defs)
    for combined_params in combined_params_iterator:
        print(dict(zip(param_names, combined_params)))


def display_named_tuple(name: str, nt: Tuple) -> None:
    r"""Displays the named tuple as a dictionary with rounded values.

    Parameters:
    -----------
    name : str
        The name of the named tuple.
    nt : Tuple
        The named tuple.
    """
    d = nt._asdict()
    _round = lambda x: round(x, 2) if isinstance(x, float) else x
    d = {k: _round(v) for k, v in d.items()}
    print(f"{name}: {d}")


def nt_values(nt: Tuple) -> Tuple:
    r"""Returns the names of the fields of the named tuple.

    Parameters:
    -----------
    nt : Tuple
        The named tuple.

    Returns:
    --------
    Tuple
        A tuple containing the names of the fields of the named tuple.
    """
    return tuple(nt._asdict().values())


def nt_names(nt: Tuple) -> Tuple:
    r"""Get the names of the fields of a named tuple.

    Parameters:
    -----------
    nt : Tuple
        The named tuple from which field names are to be extracted.

    Returns:
    --------
    Tuple
        The names of the fields of the named tuple.
    """
    return tuple(nt._asdict().keys())


def flipkart_gridsearch(
    pipeline: callable,
    corpora: Dict[str, Tuple[np.ndarray, np.ndarray]],
    corpus_name: str,
    ex_params_name: str,
    ex_param_domain_defs: Dict[str, Tuple[type, int, int, int, bool]],
    verbosity: int = 0,
    include_fixed: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""Performs a grid search on a subset of the parameters of the
    `ex_params_name` parameter of the `pipeline` function for the `corpus_name`
    corpus.
    
    Parameters
    ----------
    pipeline : callable
        The pipeline function to use for training and evaluation.
    corpora : Dict[str, Tuple[np.ndarray, np.ndarray]]
        The corpora dictionary containing the datasets to use.
    corpus_name : str
        The name of the corpus to use for training and evaluation.
    ex_params_name : str
        The name of the extractor parameters to optimize.
    ex_param_domain_defs : Dict[str, Tuple[type, int, int, int, bool]]
        A dictionary containing the domain definitions for each extractor
        parameter. The keys represent the parameter names and the values are
        tuples containing:
            - the type of the parameter value
            - the start value
            - the end value
            - the number of elements in the range
            - a boolean indicating whether to use a logarithmic scale instead
            of a linear one (optional, default=False)
    verbosity : int, optional
        The level of verbosity of the grid search (default=0).
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        A tuple containing:
            - A pandas DataFrame containing the results of the grid search.
            - A dictionary containing the fixed parameter values used during the
              grid search.
    
    Example
    -------
    >>> from gridsearch import flipkart_gridsearch
    >>> corpus_name = "product_name"
    >>> pipeline = tx_ml_word2vec
    >>> ex_params_name = 'word2vec_params'
    >>> ex_param_domain_defs = {
    >>>     "cbow": True, "vector_size": 130, "min_df": 12, "hf_thres": 63e-6,
    >>>     "window": 5, "cbow_mean":True, "hs": False, "negative": 2,
    >>>     "ns_exponent": .5, "n_iter": 5,
    >>>     "alpha": (float, 5e-3, 1e-2, 2),
    >>>     "min_alpha": (float, 5e-4, 1e-3, 2),
    >>> }
    >>> gridsearch_data, fixed_params = new_flipkart_gridsearch(
    >>>     pipeline, corpora, corpus_name,
    >>>     ex_params_name, ex_param_domain_defs,
    >>>     verbosity=1
    >>> )
    >>> cls()
    >>> display(gridsearch_data)
    >>> display("Fixed params:", fixed_params)
    """
    if include_fixed is None:
        include_fixed = []

    # Get the corpus and class labels for the specified dataset
    corpus, cla_labels = corpora[corpus_name]

    # Extract the parameter names and create an iterator over all possible
    # combinations of parameter values
    param_names = get_param_names(ex_param_domain_defs)
    combined_params_iterator = get_combined_params_iterator(ex_param_domain_defs)

    param_domains = get_param_domains(ex_param_domain_defs)
    # Separate fixed and variable parameters :
    # fixed ones are not saved in the resulting dataframe
    fixed_params = {}
    variable_param_names = []
    for param_name, param_domain in zip(param_names, param_domains):
        if len(param_domain) == 1 and param_name not in include_fixed:
            fixed_params[param_name] = param_domain[0]
        else:
            variable_param_names.append(param_name)
        # Print the parameter ranges or fixed value if verbosity is greater
        # than zero
        if verbosity > 0:
            if len(param_domain) == 1:
                print(f"Fixed '{param_name}': {param_domain[0]}")
            else:
                print(f"'{param_name}' range: {param_domain}")
    variable_param_names = tuple(variable_param_names)

    rows = []
    # Loop over all possible combinations of parameter values
    for param_values in combined_params_iterator:
        # Create a dictionary of parameter values
        params_dict = dict(zip(param_names, param_values))

        # Run the pipeline function with the specified parameters and get the
        # evaluation results
        outputs = pipeline(
            corpus, cla_labels, corpus_name,
            **{ex_params_name: params_dict},
            verbosity=verbosity
        )

        # Extract the evaluation results
        _, scores, dims, times, _ = outputs  # _, _ = labels, models

        # Print the evaluation results if verbosity is greater than zero
        if verbosity > 0:
            print("params:", params_dict)
            display_named_tuple("dims", dims)
            display_named_tuple("scores", scores)
            display_named_tuple("times", times)

        # Append the variable parameter values and evaluation results to the
        # output table
        variable_param_values = tuple(
            params_dict[param_name]
            for param_name in variable_param_names
        )
        rows.append(
            variable_param_values
            + nt_values(dims)
            + nt_values(scores)
            + nt_values(times)
        )

    if len(rows) == 0:
        return

    # Define the column names for the output table
    columns = (
        variable_param_names
        + nt_names(dims) + nt_names(scores) + nt_names(times)
    )

    # Create a Pandas DataFrame from the output table and return it with fixed
    # parameters dict
    return pd.DataFrame(rows, columns=columns), fixed_params


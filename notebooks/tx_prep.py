from typing import *
import re
import string

# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

from pepper_utils import save_and_show


""" Basics
"""


# ex. def tokenizer_fct(sentence) :
# ex. def tokenize_text(sent: str) -> List[str]:
def sent_replace_symbols(
    sent: str,
    symbols: str,
    repl: str = " "
) -> str:
    r"""Replace symbols in a string with the given replacement.

    Parameters
    ----------
    sent : str
        The input string to process.
    symbols : str
        The string of symbols to replace.
    repl : str, optional (default=" ")
        The replacement string.

    Returns
    -------
    str
        The processed string with symbols replaced by `repl`.
    """
    return re.sub(fr"([ *{symbols}] *)+", repl, sent)


def corpus_replace_symbols(
    sents: pd.Series,
    symbols: str,
    repl: str = " "
) -> pd.Series:
    r"""Replace symbols in a corpus of strings with the given replacement.

    Parameters
    ----------
    sents : pandas.Series
        The input series of strings to process.
    symbols : str
        The string of symbols to replace.
    repl : str, optional (default=" ")
        The replacement string.

    Returns
    -------
    pandas.Series
        The processed series with symbols replaced by `repl`.
    """
    return sents.str.replace(fr"([ *{symbols}] *)+", repl, regex=True)


def sent_casefold(sent: str) -> str:
    r"""Convert a string to lowercase.

    Parameters
    ----------
    sent : str
        The input string to convert.

    Returns
    -------
    str
        The string converted to lowercase.
    """
    return sent.casefold()


def corpus_casefold(sents: pd.Series) -> pd.Series:
    r"""Convert a corpus of strings to lowercase.

    Parameters
    ----------
    sents : pandas.Series
        The input series of strings to convert.

    Returns
    -------
    pandas.Series
        The series of strings converted to lowercase.
    """
    return sents.str.casefold()


def lemmatize_words(tokenized_sent: List[str]) -> List[str]:
    r"""Lemmatize words in a list using the WordNetLemmatizer from NLTK.

    Parameters
    ----------
    tokenized_sent : List[str]
        The list of words to lemmatize.

    Returns
    -------
    List[str]
        The lemmatized list of words.

    Examples
    --------
    >>> lemmatize_words(['cats', 'are', 'running'])
    ['cat', 'be', 'run']

    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokenized_sent]


def filter_words(
    tokenized_sent: List[str],
    discard_pattern: Optional[str] = None,
    stopwords: Optional[Set[str]] = None,
    isalnum: bool = False,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None
) -> List[str]:
    r"""Filter words from a tokenized sentence based on various criteria.

    Parameters
    ----------
    tokenized_sent : List[str]
        The list of tokenized words to filter.

    discard_pattern : str, optional (default=None)
        A regex pattern to match against each word. If a word matches the
        pattern, it will be discarded.

    stopwords : set, optional (default=None)
        A set of stopwords to discard. Words in this set will be discarded.

    isalnum : bool, optional (default=False)
        If True, discards words that are not alphanumeric.

    min_len : int, optional (default=None)
        If not None, discards words that are shorter than `min_len`.

    max_len : int, optional (default=None)
        If not None, discards words that are longer than `max_len`.

    Returns
    -------
    List[str]
        The filtered list of tokenized words.
    """
    if stopwords is None and not isalnum and min_len is None and max_len is None:
        return tokenized_sent
    else:
        def discard_it(word: str) -> bool:
            return (
                (discard_pattern is not None and re.match(discard_pattern, word))
                or (stopwords is not None and word in stopwords)
                or (isalnum and not word.isalnum())
                or (min_len is not None and len(word) < min_len)
                or (max_len is not None and len(word) > max_len)
            )
        return [word for word in tokenized_sent if not discard_it(word)]


def filter_tweets(tokenized_sent: List[str]) -> List[str]:
    r"""Filter tokenized sentences by removing words that start with "@" or
    "http", and removing stopwords, non-alphanumeric words, and words with
    fewer than 3 characters.
    
    Parameters
    ----------
    tokenized_sent : list of str
        List of tokenized words.
    
    Returns
    -------
    list of str
        Filtered list of tokenized words.
    """
    return filter_words(
        tokenized_sent,
        discard_pattern=r"^(@|http)",
        stopwords=stopwords.words('english'),
        isalnum=True,
        min_len=3
    )
    

def filter_product_names(tokenized_sent: List[str]) -> List[str]:
    r"""Filter tokenized sentences by removing stopwords, non-alphanumeric
    words, and words with fewer than 3 characters.
    
    Parameters
    ----------
    tokenized_sent : list of str
        List of tokenized words.
    
    Returns
    -------
    list of str
        Filtered list of tokenized words.
    """
    return filter_words(
        tokenized_sent,
        # discard_pattern=r"^(@|http)",
        stopwords=stopwords.words('english'),
        isalnum=True,
        min_len=3
    )


def filter_product_descriptions(tokenized_sent: List[str]) -> List[str]:
    r"""Filter tokenized sentences by removing stopwords, non-alphanumeric
    words, and words with fewer than 3 characters.
    
    Parameters
    ----------
    tokenized_sent : list of str
        List of tokenized words.
    
    Returns
    -------
    list of str
        Filtered list of tokenized words.
    """
    return filter_words(
        tokenized_sent,
        # discard_pattern=r"^(@|http)",
        stopwords=stopwords.words('english'),
        isalnum=True,
        min_len=3
    )


def preprocess_corpus(
    corpus: pd.Series,
    sent_tokenize=False,
    discard_symbols: str = None, 
    tokenize_func: Optional[Callable[[str], List[str]]] = word_tokenize,
    discard_words_function: Callable[[List[str]], List[str]] = None
) -> Tuple[pd.Series, pd.DataFrame]:
    r"""Preprocess a corpus of texts by applying a series of operations:
    case-folding, symbol replacement, tokenization, word filtering,
    lemmatization and rejoining.
    
    Parameters:
    -----------
    corpus : pd.Series
        The corpus of texts to preprocess.
    sent_tokenize : bool, default=False
        Whether to tokenize the sentences in the corpus. If True, the corpus
        is expected to contain multiple sentences per document, and will be
        tokenized using nltk.sent_tokenize.
    discard_symbols : str, default=None
        A string containing the symbols to remove from the texts. If None, no
        symbol is removed.
    tokenize_func : Callable[[str], List[str]], default=nltk.word_tokenize
        The function to use for tokenization.
    discard_words_function : Callable, default=None
        A function taking a list of tokens as input and returning a list of
        filtered tokens. If None, no word is filtered.
    
    Returns:
    --------
    sents : pd.Series
        The preprocessed texts.
    corpus_metadata : pd.DataFrame or None
        A DataFrame containing the metadata of the corpus (i.e., all columns
        except the last one). If `sent_tokenize` is True, returns a DataFrame
        with the same columns as the original corpus except the last one.
        Otherwise, returns None.
    """
    sents = corpus
    sent_index = None
    if sent_tokenize:
        corpus_sents = ravel_corpus_sents(corpus)
        sents = corpus_sents[corpus_sents.columns[-1]]
        sent_index = corpus_sents[corpus_sents.columns[:-1]]
    sents = corpus_casefold(sents)
    if discard_symbols is not None:
        sents = corpus_replace_symbols(sents, discard_symbols, " ")
    sents = word_tokenize_corpus(sents, tokenize_func)
    if discard_words_function is not None:
        sents = sents.apply(discard_words_function)
    sents = sents.apply(lemmatize_words)
    sents = sents.apply(lambda x: " ".join(x))
    return sents, sent_index


def apply_functions(
    series: pd.Series,
    functions: Union[Callable[[Any], Any], List[Callable[[Any], Any]]],
    result_names: Optional[List[str]] = None,
) -> Union[Tuple[pd.Series, ...], pd.Series]:
    r"""Applies one or multiple functions to the elements of a pandas Series
    and returns the results as a tuple of Series or a single Series if only one
    function was applied.

    Parameters:
    -----------
    series : pd.Series
        The Series whose elements will be processed.
    functions : Union[Callable[[Any], Any], List[Callable[[Any], Any]]]
        The function(s) to apply to the elements of the Series. It can be a
        single function or a list of functions.
    result_names : Optional[List[str]], default=None
        A list of names for the resulting Series. If None, no name is given
        to the Series.

    Returns:
    --------
    Union[Tuple[pd.Series, ...], pd.Series]
        The processed Series as a tuple of Series or a single Series if only
        one function was applied.
    """
    res = []
    if isinstance(functions, (list, tuple)):
        for f in functions:
            res.append(series.apply(f))
    else:
        res.append(series.apply(functions))
    if result_names is not None:
        for s, name in zip(res, result_names):
            s.rename(name, inplace=True)
    return tuple(res) if len(res) > 1 else res[0]


def save_lexicon(data: pd.DataFrame, name: str) -> None:
    r"""Save a DataFrame as a CSV file.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to save.
    name : str
        The name of the CSV file.

    Returns
    -------
    None
    """
    save_dir = '../tmp/tx_prep/'
    data.to_csv(save_dir + name + '.csv', encoding='utf-8', index=False)


def sent_tokenize_corpus(docs: pd.Series) -> pd.Series:
    r"""Tokenize a Series of strings using the `sent_tokenize` function
    from the NLTK package.

    Parameters
    ----------
    docs : pd.Series
        The Series of documents to tokenize.

    Returns
    -------
    pd.Series
        A Series of tokenized strings.
    """
    # return series.apply(word_tokenize)
    return apply_functions(docs, sent_tokenize)


def _number_pattern() -> str:
    r"""Returns a regular expression pattern that matches numeric values."""
    sign = r"[-+]"
    integer = r"(?:(?:\d{1,3}(?:,\d{3})+)|\d+)"
    mantissa = r"(?:\.\d+)"
    exponent = r"(?:[eE][-+]?\d+)"
    number = fr"{sign}?{integer}{mantissa}?{exponent}?"
    return number


def is_number(word: str) -> bool:
    r"""Determines whether a given string represents a numerical value.
    
    Parameters
    ----------
    word : str
        The string to be checked.
        
    Returns
    -------
    bool
        True if the string represents a numerical value, False otherwise.
    """
    pattern = re.compile(fr"^{_number_pattern()}$")
    return bool(pattern.match(word))


def is_number_with_unit(word: str) -> bool:
    r"""Determines whether a given string represents a numerical value with a
    unit.

    Parameters
    ----------
    word : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string represents a numerical value with a unit, False
        otherwise.
    """
    unit_pat = r"([a-z]{1,5}|x[a-z]+)"
    pattern = re.compile(fr"^{_number_pattern()}x?{unit_pat}$")
    return bool(pattern.match(word))


def contains_number(word: str) -> bool:
    r"""Determines whether a given string contains a numerical value.

    Parameters
    ----------
    word : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string contains a numerical value, False otherwise.
    """
    pattern = re.compile(fr"{_number_pattern()}")
    return bool(pattern.search(word))


def contains_digit(word: str) -> bool:
    r"""Determines whether a given string contains a digit.

    Parameters
    ----------
    word : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string contains a digit, False otherwise.
    """
    pattern = re.compile(r"\d")
    return bool(pattern.search(word))


def tokenize_expression_with_digit(word):
    r"""Tokenizes a string containing numerical values and alphabetic
    characters, replacing the 'x' character with '*'.

    Parameters
    ----------
    word : str
        The string to be tokenized.

    Returns
    -------
    List[str]
        A list of tokenized strings.
    """
    word = re.sub(r"(\d)x([A-Za-z])", r"\1*\2", word)
    pattern = re.compile(fr"([A-Za-z]+|{_number_pattern()})")
    return pattern.findall(word)


def strictly_tokenize_expression(word: str) -> List[str]:
    r"""Tokenizes a string containing strictly numerical or alphabetic
    characters.

    Parameters
    ----------
    word : str
        The string to be tokenized.

    Returns
    -------
    List[str]
        A list of tokenized strings.
    """
    word = re.sub(r"(\d)x([A-Za-z])", r"\1*\2", word)
    pattern = re.compile(fr"([A-Za-z]+)")
    return pattern.findall(word)


def flipkart_word_tokenize(doc: str) -> List[str]:
    r"""Tokenizes a string of text, separating words and removing numerical
    values and punctuation.
    
    Parameters
    ----------
    doc : str
        The document to tokenize.
        
    Returns
    -------
    List[str]
        The tokenized document.
    """
    words = word_tokenize(doc)
    _words = []
    for word in words:
        if contains_digit(word) or len(word) > 15:
            _words.extend(strictly_tokenize_expression(word))
        else:
            word = word.strip(" " + string.punctuation)
            if word:
                _words.append(word)
    return _words


def word_tokenize_corpus(
    docs: pd.Series,
    tokenize_func: Optional[Callable[[str], List[str]]] = word_tokenize
) -> pd.Series:
    r"""Tokenize a Series of strings using a given tokenization function.

    Parameters
    ----------
    docs : pd.Series
        The Series of documents to tokenize.
    tokenize_func : Callable[[str], List[str]], optional
        The function to use for tokenization, by default nltk.word_tokenize.

    Returns
    -------
    pd.Series
        A Series of tokenized strings.
    """
    return apply_functions(docs, tokenize_func)


def count_tag_occurrences(series: pd.Series, tag: str) -> pd.Series:
    r"""Count the number of occurrences of a tag in each element of a Series of
    strings.

    Parameters
    ----------
    series : pd.Series
        A Series of strings to search for the tag.
    tag : str
        The tag to search for.

    Returns
    -------
    pd.Series
        A Series containing the number of occurrences of the tag in each
        string.

    Examples
    --------
    >>> s = pd.Series(['apple', 'banana', 'orange'])
    >>> count_tag_occurrences(s, 'a')
    0    1
    1    3
    2    1
    dtype: int64
    """
    return series.str.count(tag)


def lens(tokens_series: pd.Series) -> pd.Series:
    r"""Compute the number of words in each sentence of a tokenized Series.

    Parameters
    ----------
    tokens_series : pd.Series
        A tokenized Series of strings representing sentences.

    Returns
    -------
    pd.Series
        A Series containing the number of words in each sentence.

    Examples
    --------
    >>> s = pd.Series(['The quick brown fox', 'jumps over', 'the lazy dog'])
    >>> lens(s)
    0    4
    1    2
    2    3
    dtype: int64
    """
    return tokens_series.apply(len)


def display_lens_dist(lens: pd.Series) -> None:
    """ DEPRECATED use `` instead
    Display the distribution of word counts in a Series of integers.

    Parameters
    ----------
    lens : pd.Series
        A Series of integers representing lengths.

    Returns
    -------
    None

    Examples
    --------
    >>> s = pd.Series([2, 3, 3, 4, 4, 4, 5, 5, 6])
    >>> display_word_count_dist(s)
          2  3  4  5  6
    freq  1  2  3  2  1
    mean: 4.00
    med : 4.00
    std : 1.31
    """
    display(pd.DataFrame(lens.value_counts().sort_index().rename('freq')).T)
    print(f"mean: {lens.mean():.2f}")
    print(f"med : {lens.median():.2f}")
    print(f"std : {lens.std():.2f}")


def display_dist(data: pd.Series, name: str = '') -> None:
    r"""Display the distribution of a Series of values.

    Parameters
    ----------
    data : pd.Series
        A Series of values to compute the distribution from.
    name : str, default=''
        A name for the data, used in the plot title.

    Returns
    -------
    None

    Examples
    --------
    >>> s = pd.Series([2, 3, 3, 4, 4, 4, 5, 5, 6])
    >>> display_dist(s, 'Word count')
        Word count distribution
    2                           1
    3                           2
    4                           3
    5                           2
    6                           1

    Mean: 4.00
    Median: 4.00
    Standard deviation: 1.31
    """
    counts = (
        data.value_counts()
        .sort_index()
        .rename_axis('Value')
        .reset_index(name='Frequency')
    )
    counts['Value'] = counts['Value'].astype(str)
    # title = f'{name} distribution' if name else 'Distribution'
    display(counts.set_index('Value').T.style.hide_index())
    print(f"\nMean: {data.mean():.2f}")
    print(f"Median: {data.median():.2f}")
    print(f"Standard deviation: {data.std():.2f}")


def show_lens_dist(
    lens: pd.Series,
    clip: Optional[Union[int, Tuple[int, int]]] = None,
    unit: Optional[str] = None,
    log_scale: Optional[Tuple[bool, bool]] = None
) -> None:
    r"""Display a histogram of the distribution of sequence lengths.

    Parameters
    ----------
    lens : pd.Series
        A Series of integers representing lengths.
    clip : int or Tuple[int, int], optional
        A tuple containing the lower and upper limits of the x-axis, or an
        integer representing the upper limit. If not specified, the x-axis
        limits are automatically set to the minimum and maximum values of
        the input sequence lengths.
    unit : str, optional
        The type of unit used in the sequence, either "elt" (default), "word"
        or "sent".
    log_scale : Tuple[bool, bool], optional
        A tuple indicating whether to use a logarithmic scale on the x-axis
        (first element) and/or y-axis (second element).

    Returns
    -------
    None

    Examples
    --------
    >>> s = pd.Series([2, 3, 3, 4, 4, 4, 5, 5, 6])
    >>> show_lens_dist(s, clip=(2, 6), unit="word", log_scale=(False, True))

    Notes
    -----
    This function generates a histogram of the distribution of sequence lengths
    in a given Pandas Series object. The x-axis shows the length of the
    sequence, and the y-axis shows the frequency of sequences with a given
    length. The input Series should contain integers representing the length of
    each sequence. If the `clip` parameter is specified, the x-axis limits are
    set to the minimum and maximum values of the input sequence lengths, or to
    the values specified in the `clip` tuple. If the `unit` parameter is
    specified, the axis labels will be updated accordingly. If the `log_scale`
    parameter is specified, the x-axis and/or y-axis will be displayed on a
    logarithmic scale.
    The function saves the plot in a PNG file in the "../img/" directory with
    the following naming convention:
    `"[unit]_len_dist_[lens.name][clip].png"`, where:
        - `[unit]` is either "elt", "sent" or "word" depending on the value of
            the `unit` parameter.
        - `[lens.name]` is the name of the input `pd.Series`.
        - `[clip]` is the value of the `clip` parameter.
    """
    if clip is None:
        clip = (1, lens.max())
    elif isinstance(clip, int):
        clip = (1, clip)
    sns.histplot(lens, binwidth=1)
    plt.xticks(range(*clip))
    plt.xlim(*clip)
    if log_scale is not None:
        log_xscale, log_yscale = log_scale
        if log_xscale:
            plt.xscale('log')
        if log_yscale:
            plt.yscale('log')
    # var_name = lens.name[:-3]
    seq_type = None
    elts_type = None
    if unit is None:
        unit = 'elt'
        seq_type = 'sequence'
        elts_type = 'elements'
    elif unit == 'sent':
        seq_type = 'document'
        elts_type = 'sentences'
    elif unit == 'word':
        seq_type = 'sentence'
        elts_type = 'words'
    plt.xlabel(f"Number of {elts_type}")
    plt.title(f"`{lens.name}` {clip} {seq_type} length distribution", weight="bold", pad=10)

    # Save and show the plot
    save_and_show(f"{unit}_len_dist_{lens.name}{clip}", sub_dir="corpora")
    """plt.savefig(
        f'../img/{unit}_len_dist_{lens.name}{clip}.png',
        facecolor='white',
        bbox_inches='tight',
        dpi=300   # x 2
    )
    plt.show()"""


def ravel_corpus_sents(docs: pd.Series) -> pd.DataFrame:
    r"""Extract sentences from a corpus and ravel them into a DataFrame.

    Parameters
    ----------
    docs : pd.Series
        A Series containing documents to tokenize.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
            - 'id': int, a unique identifier for each sentence.
            - 'sent_id': int, the sentence index within each document.
            - The text content of the sentence.

    """
    sents = sent_tokenize_corpus(docs)
    sents.index.name = 'id'
    sent_ids = sents.apply(lambda x: list(range(len(x)))).rename('sent_id')
    sents = pd.concat([sent_ids, sents], axis=1)
    sents.reset_index(inplace=True)
    return sents.explode(list(sents.columns[1:]), ignore_index=True)


def ravel_corpus_words(
    sents: pd.Series,
    tokenize_func: Optional[Callable[[str], List[str]]] = word_tokenize
) -> pd.DataFrame:
    r"""Tokenize sentences from a corpus and ravel the words into a DataFrame.

    Parameters
    ----------
    sents : pd.Series
        A Series containing sentences to tokenize.
    tokenize_func : Callable[[str], List[str]], optional
        A function used to tokenize each string in the Series. 
        If not provided, the function `word_tokenize` from the NLTK package
        will be used.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
            - 'sent_id': int, a unique identifier for each sentence.
            - 'word_id': int, the index of each word within its sentence.
            - The text content of the word.

    """
    words = word_tokenize_corpus(sents.copy(), tokenize_func)
    words.index.name = 'sent_id'
    word_ids = words.apply(lambda x: list(range(len(x)))).rename('word_id')
    words = pd.concat([word_ids, words], axis=1)
    words.reset_index(inplace=True)
    return words.explode(list(words.columns[1:]), ignore_index=True)


def ravel_corpus_sent_words(
    docs: pd.Series,
    tokenize_func: Optional[Callable[[str], List[str]]] = word_tokenize
) -> pd.DataFrame:
    r"""Extract sentences and tokenize them into words, ravel them into a
    DataFrame.

    Parameters
    ----------
    docs : pd.Series
        A Series containing documents to tokenize.
    tokenize_func : Callable[[str], List[str]], optional
        A function used to tokenize each string in the Series. 
        If not provided, the function `word_tokenize` from the NLTK package
        will be used.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
            - 'id': int, a unique identifier for each word.
            - 'sent_id': int, the sentence index within each document.
            - 'doc_sent_id': int, the sentence index within the original
                document.
            - 'word_id': int, the index of each word within its sentence.
            - The text content of the word.
            - The name of the input Series as column name for document
                identifier.

    """
    corpus_name = docs.name
    sents = ravel_corpus_sents(docs)
    words = ravel_corpus_words(sents[corpus_name], tokenize_func)
    doc_sent_ids = sents[sents.columns[:2]].copy()
    doc_sent_ids.rename(columns={'sent_id': 'doc_sent_id'}, inplace=True)
    data = pd.merge(words, doc_sent_ids, left_on='sent_id', right_index=True)
    return data[['id', 'sent_id', 'doc_sent_id', 'word_id', corpus_name]]


def get_lexicon_dict(
    sentences: pd.Series,
    tokenize_func: Optional[Callable[[str], List[str]]] = word_tokenize
) -> dict:
    """Create a dictionary of words and their frequencies from a Series of
    strings.

    Parameters
    ----------
    sentences : pd.Series
        A Series of strings to tokenize.
    tokenize_func : Callable[[str], List[str]], optional
        A function used to tokenize each string in the Series. 
        If not provided, the function `word_tokenize` from the NLTK package
        will be used.

    Returns
    -------
    dict
        A dictionary where the keys are the unique words found in the tokenized
        sentences, and the values are the frequency of each word in the
        sentences.

    Examples
    --------
    >>> s = pd.Series(['The quick brown fox', 'jumps over', 'the lazy dog'])
    >>> get_lexicon_dict(s)
    {'The': 1, 'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'the': 2, 'lazy': 1, 'dog': 1}
    """
    def addto_lexicon(
        tokenized_product_name: List[str],
        lexicon: dict
    ) -> None:
        for word in tokenized_product_name:
            if word in lexicon:
                lexicon[word] += 1
            else:
                lexicon[word] = 1

    lexicon = {}
    sentences.apply(lambda x: addto_lexicon(tokenize_func(x), lexicon))
    return lexicon


def get_lexicon_dataframe(
    sentences: pd.Series,
    tokenize_func: Optional[Callable[[str], List[str]]] = word_tokenize
) -> pd.DataFrame:
    r"""Compute a lexicon from a Series of sentences and return it as a
    DataFrame.

    Parameters
    ----------
    sentences : pd.Series
        A Series of strings representing sentences.
    tokenize_func : Callable[[str], List[str]], optional
        A function used to tokenize each string in the Series. 
        If not provided, the function `word_tokenize` from the NLTK package
        will be used.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'word', 'freq', and 'len'. 'word' contains
        the unique words in the sentences, 'freq' contains the number of times
        each word appears, and 'len' contains the length of each word.

    Examples
    --------
    >>> s = pd.Series(['The quick brown fox', 'jumps over', 'the lazy dog'])
    >>> get_lexicon_dataframe(s)
        word  freq  len
    0   The     1    3
    1 quick     1    5
    2 brown     1    5
    3   fox     1    3
    4 jumps     1    5
    5  over     1    4
    6   the     2    3
    7  lazy     1    4
    8   dog     1    3
    """
    sentences_dict = get_lexicon_dict(sentences, tokenize_func)
    data = pd.DataFrame.from_dict(sentences_dict, orient='index')
    data.reset_index(inplace=True)
    data.columns = ['word', 'freq']
    data.index.name = sentences.name
    data['len'] = data.word.apply(len)
    return data


def show_lexicon_dist(
    lexicon_data: pd.DataFrame,
    feature: str = 'len',
    clip: Optional[Union[int, Tuple[int, int]]] = None
) -> None:
    r"""Display a histogram of the values of a feature in a DataFrame.

    Parameters
    ----------
    lexicon_data : pd.DataFrame
        A DataFrame containing the data to display.
    feature : str, default='len'
        The name of the column to use for the histogram.
    clip : Union[int, Tuple[int, int]], optional
        A tuple containing the lower and upper limits of the x-axis, or an
        integer representing the upper limit. If not specified, the x-axis
        limits are automatically set to the minimum and maximum values of
        word_count.

    Returns
    -------
    None

    Notes
    -----
    This function saves the generated plot to an image file in the '../img/'
    directory.

    Examples
    --------
    >>> s = pd.Series(['The quick brown fox', 'jumps over', 'the lazy dog'])
    >>> lexicon_data = get_lexicon_dataframe(s)
    >>> show_lexicon_dist(lexicon_data, 'freq')
    """
    if clip is None:
        clip = (1, lexicon_data[feature].max())
    elif isinstance(clip, int):
        clip = (1, clip)
    sns.histplot(data=lexicon_data, x=feature, binwidth=1)
    plt.xticks(range(*clip))
    plt.xlim(*clip)
    if feature == 'freq':
        plt.xscale('log')
        plt.yscale('log')
    var_name = lexicon_data.index.name  # {clip}
    feat_name = 'length' if feature == 'len' else 'frequency'
    plt.xlabel(f"Word {feat_name}")
    plt.title(
        f"`{var_name}` {clip} word {feat_name} distribution",
        weight="bold", pad=10
    )
    # Save and show the plot
    save_and_show(f"word_{feature}_dist_{var_name}{clip}", sub_dir="corpora")
    """plt.savefig(
        f'../img/word_{feature}_dist_{var_name}{clip}.png',
        facecolor='white',
        bbox_inches='tight',
        dpi=300   # x 2
    )
    plt.show()"""


def get_casefolded_lexicon_dataframe(
    sentences: pd.Series,
    tokenize_func: Optional[Callable[[str], List[str]]] = word_tokenize
) -> pd.DataFrame:
    """Returns a DataFrame containing the casefolded words in a Series of
    sentences.

    Parameters
    ----------
    sentences : pd.Series
        The Series containing the sentences.
    tokenize_func : Callable[[str], List[str]], optional
        A function used to tokenize each string in the Series. 
        If not provided, the function `word_tokenize` from the NLTK package
        will be used.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the casefolded words and their frequency and
        length.

    Examples
    --------
    >>> s = pd.Series(['The quick brown fox', 'jumps over', 'the lazy dog'])
    >>> get_casefolded_lexicon_dataframe(s)
    	word	forms	freq	len
    0	dog	    dog	1	3
    1	fox	    fox	1	3
    2	the	    [The, the]	2	3
    3	lazy	lazy	1	4
    4	over	over	1	4
    5	brown	brown	1	5
    6	jumps	jumps	1	5
    7	quick	quick	1	5
    """
    lex_data = get_lexicon_dataframe(sentences, tokenize_func)
    casefolded_words = lex_data.word.str.casefold()
    lex_data['casefolded_word'] = casefolded_words
    lex_data.sort_values(by=['len', 'freq'], inplace=True)
    first = lambda x : x.values[0]
    scalar_or_list = lambda x: first(x) if len(x) == 1 else x
    gpby = lex_data.groupby(by='casefolded_word').agg({
        'word': scalar_or_list,
        'freq': sum,
        'len': first
    })
    gpby.index.name = 'word'
    gpby.columns = ['forms', 'freq', 'len']
    gpby.sort_values(by=['len', 'freq'], inplace=True)
    gpby.reset_index(inplace=True)
    gpby.index.name = lex_data.index.name
    return gpby


def get_flipkart_lexicon_dataframe(sents: pd.Series) -> pd.DataFrame:
    lex_data = get_casefolded_lexicon_dataframe(sents, flipkart_word_tokenize)
    words = lex_data.word
    lex_data['is_stopword_en'] = is_stopword_en(words)
    lex_data['stem'] = words.apply(lambda x: PorterStemmer().stem(x))
    lex_data['lem'] = words.apply(lambda x: WordNetLemmatizer().lemmatize(x))
    return lex_data


def get_nltk_en_stopwords() -> List[str]:
    """Return the list of English stopwords provided by the NLTK library.

    Returns
    -------
    List[str]
        A list of English stopwords.

    Examples
    --------
    >>> get_nltk_en_stopwords()[:5]
    ['i', 'me', 'my', 'myself', 'we']
    """
    return stopwords.words('english')


def casefold_words(words: pd.Series) -> pd.Series:
    r"""Convert the characters in a Series of strings to their casefolded form.

    Parameters
    ----------
    words : pd.Series
        A Series of strings.

    Returns
    -------
    pd.Series
        A Series of strings with characters in casefolded form.

    Examples
    --------
    >>> s = pd.Series(['The Quick Brown Fox', 'jumps over', 'the Lazy Dog'])
    >>> casefold_words(s)
    0    the quick brown fox
    1              jumps over
    2            the lazy dog
    dtype: object
    """
    return words.str.casefold()


def is_stopword_en(words: pd.Series) -> pd.Series:
    r"""Check whether the words in a Series of strings are English stopwords.

    Parameters
    ----------
    words : pd.Series
        A Series of strings.

    Returns
    -------
    pd.Series
        A Series of boolean values indicating whether each word is an English
        stopword.

    Examples
    --------
    >>> s = pd.Series(['i', 'am', 'not', 'a', 'stopword'])
    >>> is_stopword_en(s)
    0     True
    1     True
    2     True
    3     True
    4    False
    dtype: bool
    """
    return words.isin(get_nltk_en_stopwords())


def show_n_grams(lexicon_data: pd.DataFrame, n: int) -> None:
    r"""Display all n-grams of length n in a DataFrame of word frequencies.

    Parameters
    ----------
    lexicon_data : pd.DataFrame
        A DataFrame containing columns 'word', 'freq', and 'len'.
    n : int
        The length of the n-grams to display.

    Returns
    -------
    None

    Examples
    --------
    >>> s = pd.Series(['apple', 'banana', 'orange'])
    >>> lexicon_data = get_lexicon_dataframe(s)
    >>> show_n_grams(lexicon_data, 3)
    nombre de 3-grammes : 2
    array(['ang', 'ban'], dtype=object)

    """   
    n_grams = lexicon_data[lexicon_data.len == n]
    print(f'{n}-grams count:', n_grams.shape[0])
    n_grams_ = n_grams.word.values.copy()
    n_grams_.sort()
    display(n_grams_)
    display(n_grams.sort_values(by='freq', ascending=False))


def is_integer(words: pd.Series) -> pd.Series:
    r"""Check if each word in a Series is an integer.

    Parameters
    ----------
    words : pd.Series
        A Series of strings.

    Returns
    -------
    pd.Series
        A boolean Series indicating whether each word is an integer.

    Examples
    --------
    >>> s = pd.Series(['123', '1a', '5', '10'])
    >>> is_integer(s)
    0     True
    1    False
    2     True
    3     True
    dtype: bool

    """
    return words.str.match(r'\d+')


def is_upper_alpha(words: pd.Series) -> pd.Series:
    r"""Check if each word in a Series contains only uppercase alphabetic
    characters.

    Parameters
    ----------
    words : pd.Series
        A Series of strings.

    Returns
    -------
    pd.Series
        A boolean Series indicating whether each word contains only uppercase
        alphabetic characters.

    Examples
    --------
    >>> s = pd.Series(['ABC', 'abC', 'DEF', '123'])
    >>> is_upper_alpha(s)
    0     True
    1    False
    2     True
    3    False
    dtype: bool

    """
    return words.str.match(r'[A-Z]+')


def compute_text_metrics(text: str) -> dict:
    r"""Compute text metrics for a given text.

    Parameters
    ----------
    text : str
        The text to analyze.

    Returns
    -------
    dict
        A dictionary containing the following metrics:
        - 'char_count': the total number of characters in the text
        - 'word_count': the total number of words in the text
        - 'sent_count': the total number of sentences in the text
        - 'avg_word_length': the average length of a word in the text
        - 'avg_sent_length': the average length of a sentence in the text
        - 'avg_sent_complexity': the average number of words per sentence
            in the text

    Examples
    --------
    >>> text = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore."
    >>> compute_text_metrics(text)
    {'char_count': 67,
     'word_count': 13,
     'sent_count': 2,
     'avg_word_length': 4.846153846153846,
     'avg_sent_length': 6.5,
     'avg_sent_complexity': 6.5}
    """
    import re
    
    # Compute character count
    char_count = len(text)
    
    # Compute word count
    word_count = len(re.findall(r'\w+', text))
    
    # Compute sentence count
    sent_count = len(re.findall(r'[.!?]+', text))
    
    # Compute average word length
    words = re.findall(r'\w+', text)
    avg_word_length = sum(len(word) for word in words) / word_count
    
    # Compute average sentence length
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    avg_sent_length = sum(len(sentence.split()) for sentence in sentences) / sent_count
    
    # Compute average sentence complexity
    avg_sent_complexity = word_count / sent_count
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sent_count': sent_count,
        'avg_word_length': avg_word_length,
        'avg_sent_length': avg_sent_length,
        'avg_sent_complexity': avg_sent_complexity
    }

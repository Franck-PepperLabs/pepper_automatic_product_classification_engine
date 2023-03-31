from typing import *
import pandas as pd
from IPython.display import display
from nltk.tokenize import word_tokenize


def depr_3k_sample(raw_tweets: pd.DataFrame) -> pd.DataFrame:
    """
    Selects 1500 positive and 1500 negative tweets from the input DataFrame.

    DEPRECATED, `tx_ex` original version, use `tweets_3k_sample` instead.

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


def depr_encode_cats(sentiment: pd.Series) -> Tuple[pd.Series, List[str]]:
    """Encode the sentiment categories into numerical values.

    DEPRECATED, `tx_ex` original version, use `tx_ml.encode_cats` instead.

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


def depr_count_words(raw_sent: str) -> int:
    """Count the number of words in a sentence string.

    DEPRECATED:
    * in original example : old_tweets.sent_bow.apply(old_count_words)
    * use `tx_prep.word_counts(old_tweets)` instead

    Args:
    raw_sent (str): The sentence string.

    Returns:
    int: The count of words in `raw_sent`.
    """
    return len(word_tokenize(raw_sent))


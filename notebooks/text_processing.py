from typing import *

# import nltk
from nltk.corpus import stopwords

# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display


def get_word_count(sentences):
    return sentences.apply(len).rename(sentences.name + '_wc')


def display_word_count_dist(word_count):
    display(pd.DataFrame(word_count.value_counts().sort_index().rename('freq')).T)


def show_word_count_dist(word_count, clip=None):
    if clip is None:
        clip = (1, word_count.max())
    elif isinstance(clip, int):
        clip = (1, clip)
    sns.histplot(word_count, binwidth=1)
    plt.xticks(range(*clip))
    plt.xlim(*clip)
    plt.show()


def get_lexicon_dict(sentences: pd.Series) -> dict:

    def addto_lexicon(tokenized_product_name, lexicon):
        for word in tokenized_product_name:
            if word in lexicon:
                lexicon[word] += 1
            else:
                lexicon[word] = 1

    lexicon = {}
    sentences.apply(lambda x: addto_lexicon(x, lexicon))
    return lexicon


def get_lexicon_dataframe(sentences: pd.Series) -> pd.DataFrame:
    sentences_dict = get_lexicon_dict(sentences)
    data = pd.DataFrame.from_dict(sentences_dict, orient='index')
    data.reset_index(inplace=True)
    data.columns = ['word', 'freq']
    data.index.name = sentences.name
    data['len'] = data.word.apply(len)
    return data


def show_lexicon_dist(lexicon_data, feature='len'):
    sns.histplot(data=lexicon_data, x=feature, binwidth=1)
    plt.show()


def get_nltk_en_stopwords():
    return stopwords.words('english')
    #stop_w = list(set())
    #+ ['[', ']', ',', '.', ':', '?', '(', ')']


def casefold_words(words):
    return words.str.casefold()


def is_stopword_en(words):
    return words.isin(get_nltk_en_stopwords())


def show_n_grams(lexicon_data, n):
    n_grams = lexicon_data[lexicon_data.len == n]
    print(f'nombre de {n}-grammes :', n_grams.shape[0])
    n_grams_ = n_grams.word.values.copy()
    n_grams_.sort()
    display(n_grams_)
    display(n_grams.sort_values(by='freq', ascending=False))


def is_integer(words):
    return words.str.match(r'\d+')

def is_upper_alpha(words):
    return words.str.match(r'[A-Z]+')


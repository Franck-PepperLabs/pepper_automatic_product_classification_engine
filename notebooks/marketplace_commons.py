from typing import *
from nltk.tokenize import sent_tokenize, word_tokenize

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython.display import display


def _load_data():
    filepath = '../dataset/Flipkart/flipkart_com-ecommerce_sample_1050.csv'
    data = pd.read_csv(filepath, encoding='utf-8', index_col='uniq_id')
    data.index.name = 'Flipkart'
    return data


_data = _load_data()


def save_dataframe(data, name):
    save_dir = '../dataset/save/'
    data.to_csv(save_dir + name + '.csv', encoding='utf-8', index=False)


def get_raw_data(name=None):
    global _data
    if name is None:
        return _data.copy()
    return _data[name].copy()


def full_show_sample(series, n=5):
    sample = series.sample(n)
    for item in sample.items():
        display(item)


""" Text processing
"""


def get_product_category_branches() -> pd.DataFrame:
    cats = (
        get_raw_data('product_category_tree')
        .str.replace('\["|"\]', '', regex=True)
        .str.split(' >> ', regex=True, expand=True)
    )
    cats.columns = [f'level_{i}' for i in cats.columns]
    return cats


"""tokenized_branches = (
    product_category_tree
    .str.replace('\["|"\]', '', regex=True)
    .str.split(' >> ', regex=True, expand=True)
)"""


def get_raw_tokenized_product_names():
    return (
        get_raw_data('product_name')
        .apply(word_tokenize)
    )


""" Image processing
"""


def display_product_gallery(df, size='small'):
    img_path = '../dataset/Flipkart/Images/'

    # Define the number of columns based on the thumbnail size
    if size == 'small':
        ncols = 8
    elif size == 'middle':
        ncols = 4
    elif size == 'big':
        ncols = 2
    else:
        raise ValueError('Invalid value for size_vignette')
    
    # Calculate the number of rows needed
    nrows = np.ceil(len(df) / ncols).astype(int)
    
    # Initialize the figure and grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(20, nrows * (20 / ncols))
    )
    axes = axes.ravel()
    
    # Hide the axes for each subplot
    for ax in axes:
        ax.axis('off')
    
    # Display the images, product name, and file name
    for i, (product, file) in enumerate(df[['product_name', 'image']].values):
        # TODO : fonction pour justifier le titre
        axes[i].set_title(product[:15] + '\n' + product[15:30], fontsize=15, fontweight='bold')
        img = mpimg.imread(img_path + file)
        axes[i].imshow(img)
        axes[i].text(
            0.5, -0.2,
            f'{img.shape[0]}x{img.shape[1]} pixels',
            ha='center', fontsize=8,
            transform=axes[i].transAxes
        )
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=1.2)
    plt.show()
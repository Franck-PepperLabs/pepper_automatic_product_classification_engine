from typing import *
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython.display import display


def _load_data() -> pd.DataFrame:
    r"""Load the Flipkart dataset.

    Returns
    -------
    pandas.DataFrame
        The Flipkart dataset.
    """
    filepath = '../dataset/Flipkart/flipkart_com-ecommerce_sample_1050.csv'
    data = pd.read_csv(filepath, encoding='utf-8', index_col='uniq_id')
    data.index.name = 'Flipkart'
    return data


_data = _load_data()


def get_raw_data(
    name: Optional[str] = None
) -> Union[pd.DataFrame, pd.Series]:
    r"""Return the raw data from the Flipkart dataset.

    Parameters
    ----------
    name : str, optional
        If specified, return only the column with the given name.

    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        The raw data from the Flipkart dataset.
        If `name` is None, return the entire dataset as a DataFrame.
        If `name` is not None, return only the specified column as a Series.
    """
    global _data
    if name is None:
        return _data.copy()
    return _data[name].copy()



def full_show_sample(
    series: pd.Series,
    n: Optional[int] = 5,
    print_func: Callable = display
) -> None:
    r"""Print a sample of a pandas Series.

    Parameters
    ----------
    series : pd.Series
        The series to display.
    n : int, optional
        The number of items to display. If None, display the entire series.
        Default is 5.
    print_func : Callable, optional
        The printing function to use. Default is `display`.

    Returns
    -------
    None
    """
    sample = None
    if n is None:
        sample = series
    else: 
        sample = series.sample(n)
    for item in sample.items():
        print_func(item)


""" Text processing
"""


def get_product_category_branches() -> pd.DataFrame:
    r"""Extracts product categories from the `product_category_tree` column
    of the raw data and creates a new DataFrame with separate columns for each
    level of category.

    Returns:
        pd.DataFrame: A DataFrame containing separate columns for each level of
        category. The index contains the original IDs from the
        `product_category_tree` column.
    """
    start_tag = '\\["'
    end_tag = '"\\]'
    sep_tag = " >> "
    cats = (
        get_raw_data("product_category_tree")
        .str.replace(f"{start_tag}|{end_tag}", "", regex=True)
        .str.split(sep_tag, regex=True, expand=True)
    )
    cats.index.name = "id"
    cats.columns = [f"level_{i}" for i in cats.columns]
    return cats


def get_product_categories(depth: int = 1) -> pd.DataFrame:
    r"""Get the product categories as a DataFrame up to a given depth.

    Parameters:
    -----------
    depth: int
        The maximum depth to keep. Defaults to 1.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with categories up to the given depth.
    """
    cats = get_product_category_branches()
    cats = cats[cats.columns[:depth]]
    return cats.astype("category")


def get_class_labels() -> pd.Series:
    r"""Get the class labels for each product.

    Returns:
    --------
    pd.Series
        A Series with integer codes for each class.
    """
    cats = get_product_categories()
    cats = cats[cats.columns[0]]
    return cats.cat.codes.rename("cla")


def get_class_names() -> pd.Series:
    r"""Get the class names for each product.

    Returns:
    --------
    pd.Series
        A Series with strings for each class.
    """
    cats = get_product_categories()
    return cats[cats.columns[0]].astype(str)



def get_class_label_name_map() -> Dict[int, str]:
    r"""Get a mapping between class labels and class names.

    Returns:
    --------
    Dict[int, str]
        A dictionary mapping class labels to class names.
    """
    return dict(enumerate(
        get_product_categories().level_0.cat.categories
    ))

""" Products specs
"""


def extract_key_value_pair(x):
    if x is None or x == '}':
        return None
    if x.startswith('"value"=>'):
        return ('_', x[10:-1])
    m = re.match(r'"key"=>"(.*)", "value"=>"(.*)"', x)
    if m is None:
        print('parsing error :', x)
        return None
    return m.group(1), m.group(2)


def tokenize_specs(raw_specs):
    split_specs = (
        raw_specs
        .str.replace(r'({"product_specification"=>\[?{?(?:nil)?|}?\]?}^$)', '', regex=True)
        .str.split(r'}, {', regex=True, expand=True)
    )
    return split_specs.applymap(extract_key_value_pair)


def tokenized_specs_to_dicts(tokenized_specs):
    new_specs = pd.Series(dtype=object, name='dict_specs')
    for row in tokenized_specs.iterrows():
        id = row[0]
        row_ = row[1]
        d = {}
        for pair in row_:
            if pair is None:
                break
            k, v = pair
            if k not in d:
                d[k] = v
            else:
                d[k] = [d[k]] + [v]
        new_specs[id] = d
    return new_specs
    

def get_dict_specs():
    specs = get_raw_data('product_specifications')
    nil_spec = '{"product_specification"=>nil}'
    specs[specs.isna()] = nil_spec
    tokenized_specs = tokenize_specs(specs)
    return tokenized_specs_to_dicts(tokenized_specs)


def get_key_freqs(dict_specs):
    all_keys = {}
    for _, spec in dict_specs.items():
        #id = row[0]
        #row_ = row[1]
        for k in spec.keys():
            if k in all_keys:
                all_keys[k] += 1
            else:
                all_keys[k] = 1
    return (
        pd.DataFrame.from_dict(all_keys, orient='index', columns=['freq'])
        .sort_values(by='freq', ascending=False)
    )


def get_top_specs(
        specs: pd.Series,
        keys_freqs: pd.DataFrame,
        thres: int = 0
) -> pd.DataFrame:
    columns = keys_freqs[keys_freqs.freq > thres].index
    data = pd.DataFrame(index=specs.index, columns=columns)
    for col in columns:
        data[col] = specs.apply(lambda x: x.get(col))
    return data


""" Note :
import timeit
n_iter = 100
print("get_top_specs_v1:", timeit.timeit(lambda: get_top_specs_v1(specs, keys_freqs, thres), number=n_iter))
print("get_top_specs_v2:", timeit.timeit(lambda: get_top_specs_v2(specs, keys_freqs, thres), number=n_iter))

> get_top_specs_v1: 51.95563810004387
> get_top_specs_v2: 4.504717299947515

def get_top_specs_v1(
        specs: pd.Series,
        keys_freqs: pd.DataFrame,
        thres: int = 0
) -> pd.DataFrame:
    columns = keys_freqs[keys_freqs.freq > thres].index
    data = pd.DataFrame(index=specs.index, columns=columns)
    # remplissage
    for id, spec in specs.items():
        relevant_keys = list(set(columns) & set(spec.keys()))
        for k in relevant_keys:
            data.loc[id, k] = spec[k]
    return data
"""




""" Image processing
"""


def display_product_gallery(df: pd.DataFrame, size: str = 'small') -> None:
    r"""Display a gallery of product images along with their name and size.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the products to display. The DataFrame 
        must have the columns 'product_name' and 'image', which contains the 
        name of the product and the name of the image file, respectively.
    size : str, optional (default='small')
        The size of the thumbnails. Must be one of 'small', 'middle', or 'big'.
    
    Raises
    ------
    ValueError
        If `size` is not one of 'small', 'middle', or 'big'.
    """
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
    _, axes = plt.subplots(
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
        axes[i].set_title(
            product[:15] + '\n' + product[15:30],
            fontsize=15, fontweight='bold'
        )
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

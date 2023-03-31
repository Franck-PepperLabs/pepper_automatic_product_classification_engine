from typing import *
import warnings
import os, glob
import math
import numpy as np
import pandas as pd
from IPython.display import display

import cv2 as cv

import PIL.Image as pil
import PIL.ImageDraw as draw
from PIL import ImageOps
from PIL.ImageOps import autocontrast, equalize

# from PIL.Image import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpi
# from matplotlib.axes import Axes
import seaborn as sns
# from scipy.signal import convolve2d, correlate2d

import skimage as ski
import skimage.feature as skif  # TODO : à revoir
from skimage import filters, exposure


#from skimage import color
#from skimage.color import (
#    rgb2hsv                  # 3.2. RGB to HSV
#)

from pepper_utils import (
    create_if_not_exist,
    for_all,
    get_file_size,
    format_iB,
    get_start_time,
    print_time_perf,
    save_and_show
)

""" Dir scanning
"""


def _get_filenames_scandir(
    root_dir: str,
    ext: Optional[str] = None,
    recursive: bool = False
) -> List[str]:
    """
    Returns a list of filenames in the specified directory.

    Args:
        root_dir (str):
            The root directory to search for filenames in.
        ext (str, optional):
            The extension to filter the filenames by.
            Defaults to None.
        recursive (bool, optional):
            Whether or not to search for filenames recursively.
            Defaults to False.

    Returns:
        List[str]: A list of filenames found in the directory.
    """
    filenames = []
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.name.startswith('.'):
                pass
            elif entry.is_file():
                if ext is None or entry.name.endswith(f".{ext}"):
                    filenames.append(entry.name)
            elif recursive and entry.is_dir():
                children = _get_filenames_scandir(
                    root_dir + entry.name,
                    ext, recursive
                )
                children = [
                    entry.name + "/" + filename
                    for filename in children
                ]
                filenames.extend(children)
    return filenames


def _get_filenames_glob(
    root_dir: str,
    ext: Optional[str] = None,
    recursive: bool = False
) -> List[str]:
    """
    Returns a list of filenames in the specified directory
    using glob pattern matching.

    Args:
        root_dir (str):
            The root directory to search for filenames in.
        ext (str, optional):
            The extension to filter the filenames by.
            Defaults to None, which returns all files.
        recursive (bool, optional):
            Whether or not to search for filenames recursively.
            Defaults to False.

    Returns:
        List[str]:
            A list of filenames found in the directory.
    """
    ext = "*" if ext is None else ext
    if recursive:
        filenames = glob.glob(f"**/*.{ext}", root_dir=root_dir, recursive=True)
    else:
        filenames = glob.glob(f"*.{ext}", root_dir=root_dir)
    filenames = [filename.replace("\\", "/") for filename in filenames]
    return filenames


def get_filenames(
    root_dir: str,
    ext: Optional[str] = None,
    recursive: bool = False,
    meth: str = 'glob'
) -> List[str]:
    """
    Returns a list of filenames in the specified directory
    using the specified method.

    Args:
        root_dir (str):
            The root directory to search for filenames in.
        ext (str, optional):
            The extension to filter the filenames by.
            Defaults to None, which returns all files.
        recursive (bool, optional):
            Whether or not to search for filenames recursively.
            Defaults to False.
        meth (str, optional):
            The method to use for finding filenames.
            Can be either 'glob' or 'scandir'.
            Defaults to 'glob'.

    Returns:
        List[str]:
            A list of filenames found in the directory.
    """
    if meth == 'scandir':
        return _get_filenames_scandir(root_dir, ext, recursive)
    return _get_filenames_glob(root_dir, ext, recursive)


def get_file_names_and_ids(
    root_dir: str,
    ext: Optional[str] = None,
    recursive: bool = False,
    meth: str = 'glob'
) -> Tuple[List[str], List[str]]:
    """
    Returns a tuple containing two lists:
    the list of filenames found in the specified directory,
    and the list of corresponding file ids.

    Args:
        root_dir (str):
            The root directory to search for filenames in.
        ext (str, optional):
            The extension to filter the filenames by.
            Defaults to None, which returns all files.
        recursive (bool, optional):
            Whether or not to search for filenames recursively.
            Defaults to False.
        meth (str, optional):
            The method to use for finding filenames.
            Can be either 'glob' or 'scandir'.
            Defaults to 'glob'.

    Returns:
        Tuple[List[str], List[str]]:
            A tuple containing two lists:
            the list of filenames found in the directory,
            and the list of corresponding file ids.
    """
    filenames = get_filenames(root_dir, ext, recursive, meth)
    fileids = [filename[-36:-4] for filename in filenames]
    return filenames, fileids


""" Convert
"""

# def img_to_imx(img):
def img_to_imx(im: Union[np.ndarray, pil.Image]) -> np.ndarray:
    """
    Convert a numpy array or a PIL Image to a numpy array.

    Args:
        img (Union[np.ndarray, pil.Image]):
            The image to convert.

    Returns:
        np.ndarray:
            The converted image as a numpy array.
    """
    if isinstance(im, np.ndarray):
        return im
    return np.array(im)


# def imx_to_img(imx):
def imx_to_img(im: Union[np.ndarray, pil.Image]) -> pil.Image:
    """
    Convert a numpy array to a PIL Image.

    Args:
        im (Union[np.ndarray, pil.Image]):
            The numpy array to convert.

    Returns:
        PIL.Image:
            The converted image as a PIL Image.
    """
    if isinstance(im, pil.Image):
        return im
    return pil.fromarray(im)


""" Image loading
"""


def _load_image_mpi_imread(
    filename: str,
    root_dir: Optional[str] = None
) -> np.ndarray:
    """
    Loads an image file using `matplotlib.image.imread`
    and returns it as a NumPy array.

    Args:
        filename (str):
            The name of the file to load.
        root_dir (str, optional):
            The root directory to use when loading the file. Defaults to None.

    Returns:
        np.ndarray:
            The loaded image as a NumPy array.
    """
    filepath = filename
    if root_dir is not None:
        filepath = root_dir + filename
    return mpi.imread(filepath)


def _load_image_ocv_imread(
    filename: str,
    root_dir: Optional[str] = None
) -> np.ndarray:
    """
    Loads an image file using OpenCV's `cv2.imread`
    and returns it as a NumPy array.

    Args:
        filename (str):
            The name of the file to load.
        root_dir (str, optional):
            The root directory to use when loading the file.
            Defaults to None.

    Returns:
        np.ndarray:
            The loaded image as a NumPy array.
    """
    filepath = filename
    if root_dir is not None:
        filepath = root_dir + filename
    return cv.imread(filepath)


def _load_image_pil_open(
    filename: str,
    root_dir: Optional[str] = None
) -> Tuple[np.ndarray, pil.Image]:
    """
    Loads an image file using PIL's `Image.open`
    and returns it as both a NumPy array and a PIL Image object.

    Args:
        filename (str):
            The name of the file to load.
        root_dir (str, optional):
            The root directory to use when loading the file.
            Defaults to None.

    Returns:
        Tuple[np.ndarray, pil.Image]:
            A tuple containing the loaded image
            as a NumPy array and a PIL Image object.
    """
    filepath = filename
    if root_dir is not None:
        filepath = root_dir + filename
    img = pil.open(filepath)
    imx = np.array(img)
    return imx, img


def load_image(
    filename: str,
    root_dir: Optional[str] = None,
    meth: str = 'mpi'
) -> Union[np.ndarray, Tuple[np.ndarray, pil.Image]]:
    """
    Loads an image file using one of three possible methods and returns
    it as a NumPy array or a tuple of a NumPy array and a PIL Image object.

    Args:
        filename (str):
            The name of the file to load.
        root_dir (str, optional):
            The root directory to use when loading the file.
            Defaults to None.
        meth (str, optional):
            The method to use when loading the file.
            Possible values are 'mpi' (default), 'ocv', and 'pil'.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, pil.Image]]:
            The loaded image as a NumPy array or a tuple
            of a NumPy array and a PIL Image object.
    """
    if meth == 'mpi':
        return _load_image_mpi_imread(filename, root_dir)
    if meth == 'ocv':
        return _load_image_ocv_imread(filename, root_dir)
    if meth == 'pil':
        return _load_image_pil_open(filename, root_dir)


""" Save
"""


def is_float_0_1(imx: np.ndarray) -> bool:
    """Check if the input array contains only float values between 0 and 1.

    Args:
        imx (np.ndarray):
            The input array to check.

    Returns:
        bool:
            True if the array only contains float values between 0 and 1,
            False otherwise.
    """
    return 0.0 <= np.min(imx) <= np.max(imx) <= 1.0


def from_float_0_1_to_int_0_255(imx: np.ndarray) -> np.ndarray:
    """Convert an input array containing float values between 0 and 1
    to an array containing integers between 0 and 255.
    
    Args:
        imx (np.ndarray):
            The input array to convert.

    Returns:
        np.ndarray:
            The converted array.
    """
    return (imx * 255).astype(np.uint8)


def _pil_save(
    img: pil.Image,
    filename: str,
    root_dir: Optional[str] = None
) -> None:
    """Save a PIL Image to disk.

    Args:
        img (PIL.Image):
            The PIL Image to save.
        filename (str):
            The filename to use when saving the image.
        root_dir (str, optional):
            The root directory where the image should be saved.
            Defaults to None.
    """
    filepath = filename
    if root_dir is not None:
        filepath = root_dir + filename
    img.save(filepath)


def save_hsv_as_npy(
    imx: np.ndarray,
    filename: str,
    root_dir: Optional[str] = None
) -> None:
    """Save an HSV image as a NumPy array to disk.

    Args:
        imx (numpy.ndarray):
            The input array to save.
        filename (str):
            The filename to use when saving the array.
        root_dir (str, optional):
            The root directory where the array should be saved. Defaults to None.
    """
    filepath = filename
    if root_dir is not None:
        filepath = root_dir + filename
    np.save(filepath, imx)    


def load_hsv_from_npy(
    filename: str,
    root_dir: Optional[str] = None
) -> np.ndarray:
    """Load an array of the HSV values of an image from a .npy file.

    Args:
        filename (str):
            The filename of the .npy file.
        root_dir (Optional[str], optional):
            The root directory where the file is located.
            Defaults to None.

    Returns:
        np.ndarray:
            The array of HSV values of the image.
    """
    filepath = filename
    if root_dir is not None:
        filepath = root_dir + filename
    return np.load(filepath)


def save_image(
    im: Union[np.ndarray, pil.Image],
    filename: str,
    root_dir: Optional[str] = None,
    meth: str = 'pil'
) -> None:
    """Save an image to a file.

    Args:
        im (Union[np.ndarray, pil.Image]):
            The image to save.
        filename (str):
            The filename of the file to save the image to.
        root_dir (Optional[str], optional):
            The root directory where the file will be saved.
            Defaults to None.
        meth (str, optional):
            The method to use to save the image.
            Defaults to 'pil'.
    """
    imx = img_to_imx(im)
    if is_float_0_1(imx):
        imx = from_float_0_1_to_int_0_255(imx)
    if meth == 'pil':
        _pil_save(imx_to_img(imx), filename, root_dir)
    elif meth == 'mpi':
        mpi.imsave(os.path.join(root_dir, filename), imx)
    elif meth == 'ocv':
        cv.imwrite(os.path.join(root_dir, filename), imx)


def save_all(
    ims: List[np.ndarray],
    filenames: List[str],
    root_dir: Union[None, str] = None,
    meth: str = 'pil'
) -> None:
    """
    Saves a list of images to disk with the corresponding filenames.

    Args:
        ims (List[numpy.ndarray]):
            A list of images represented as ndarrays.
        filenames (List[str]):
            A list of filenames corresponding to each image.
        root_dir (Union[None, str], optional):
            The root directory where the images will be saved.
            Defaults to None.
        meth (str, optional):
            The method used to save the images.
            Defaults to 'pil'.

    Returns:
        None
    """
    if root_dir is not None:
        create_if_not_exist(root_dir)
    for_all(
        save_image,
        args_vect=zip(ims, filenames),
        const_args=(root_dir, meth)
    )


""" Grayscale
"""

def _to_gray_ski(
    im: Union[np.ndarray, pil.Image]
) -> np.ndarray:
    """Convert the given image to grayscale using skimage.

    Args:
        im (Union[numpy.ndarray, PIL.Image.Image]:
            The image to convert.

    Returns:
        numpy.ndarray:
            The converted grayscale image.
    """
    return ski.color.rgb2gray(img_to_imx(im))


def _to_gray_pil(
    im: Union[np.ndarray, pil.Image]
) -> pil.Image:
    """Convert the given image to grayscale using PIL.

    Args:
        im (Union[numpy.ndarray, PIL.Image.Image]:
            The image to convert.

    Returns:
        PIL.Image.Image:
            The converted grayscale image.
    """
    return ImageOps.grayscale(imx_to_img(im))


def _to_gray_ocv(
    im: Union[np.ndarray, pil.Image]
) -> np.ndarray:
    """Convert the given image to grayscale using OpenCV.

    Args:
        im (Union[numpy.ndarray, PIL.Image.Image]:
            The image to convert.

    Returns:
        numpy.ndarray:
            The converted grayscale image.
    """
    return cv.cvtColor(img_to_imx(im), cv.COLOR_BGR2GRAY)


def to_gray(im, meth='ocv'):
    if meth == 'ocv':
        return _to_gray_ocv(im)
    if meth == 'ski':
        return _to_gray_ski(im)
    if meth == 'pil':
        return img_to_imx(_to_gray_pil(im))


def gray_all(ims):
    return for_all(to_gray, ims)


""" Resize
"""

def get_resize_ratio(imx):
    dy, dx = imx.shape[:2]
    size_px = dx * dy
    if size_px < 1_000_000:
        return 1.0
    return 1_000 / math.sqrt(dx * dy)


def _resize_ocv(im, interpolation=cv.INTER_AREA):
    imx = img_to_imx(im)
    r = get_resize_ratio(imx)
    return cv.resize(imx, dsize=None, fx=r, fy=r, interpolation=interpolation)


def resize_image(im, meth='ocv'):
    return _resize_ocv(im, interpolation=cv.INTER_LANCZOS4)


def resize_all(ims):
    return for_all(resize_image, ims)


""" Contrast and equalization
"""


def _contrast_pil(img):
    return equalize(autocontrast(imx_to_img(img)))


def _contrast_ocv(imx):
    imx = img_to_imx(imx)
    gray_imx = cv.cvtColor(imx, cv.COLOR_BGR2GRAY)
    return cv.equalizeHist(gray_imx)


def _contrast_ski(imx):
    imx = img_to_imx(imx)
    p2, p98 = np.percentile(imx, (2, 98))
    return exposure.rescale_intensity(imx, in_range=(p2, p98))


def contrast(im, meth='ski'):
    if meth == 'ocv':
        return _contrast_ocv(im)
    if meth == 'ski':
        return _contrast_ski(im)
    if meth == 'pil':
        return img_to_imx(_contrast_pil(im))


def contrast_all(ims):
    return for_all(contrast, ims)


""" Show image.s
"""


def show_gray_imx(gray_imx):
    plt.imshow(gray_imx, cmap="Greys_r")
    plt.show()


def show_imxs_gallery(imxs, ids, cmap=None):
    n_imxs = len(imxs)
    n_cols = 4
    n_rows = math.ceil(n_imxs / n_cols)
    _, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    ax = axes.ravel()

    for i, (id, imx) in enumerate(zip(ids, imxs)):
            ax[i].set_title(id, fontsize=5)  #, fontsize=15, fontweight='bold')
            ax[i].imshow(imx, cmap=cmap)
            ax[i].set_axis_off()

    for i in range(n_imxs, n_rows * n_cols):
            ax[i].set_axis_off()        

    plt.tight_layout()
    #plt.show()

    save_and_show(f"gallery_{len(ids)}", sub_dir="show_imxs_gallery")



def advanced_show_imx(
    imx,
    ax=None,
    title=None,
    cmap=None,
    vmin=None,
    vmax=None,
    interpolation=None,
    extent=None,
    show_grid=True,
    major_grid_step=None,
    major_grid_linestyle='-',
    minor_grid_step=None,
    minor_grid_linestyle=':',
    bg_color='black',
    bg_alpha=1,
    fg_color='white',
    grid_color=None
):
    x_min, x_max, y_max, y_min = (
        extent if extent is not None
        else (0, imx.shape[1] - 1, imx.shape[0] - 1, 0)
    )
    #print(x_min, x_max, y_max, y_min)
    w = x_max - x_min + 1
    h = y_max - y_min + 1

    cmap = 'gray' if cmap is None and imx.ndim == 2 else cmap
    vmin = 0 if vmin is None else vmin
    vmax = 255 if vmax is None else vmax

    # See : https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
    text_color = fg_color
    tick_color = fg_color
    grid_color = grid_color if grid_color is not None else fg_color

    if ax is None:
        fig, ax = plt.subplots(1, 1)

        # set figure facecolor
        fig.patch.set_facecolor(bg_color)
        fig.patch.set_alpha(bg_alpha)

    ax.imshow(imx, interpolation=interpolation, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

    # set tick and ticklabel color
    ax.tick_params(color=tick_color, labelcolor=text_color)

    if show_grid:
        # See https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels-in-matplotlib
        minor_step = (
            minor_grid_step if minor_grid_step is not None
            else math.floor(math.sqrt((w / 25) * (h / 25)))
        )
        if minor_step < 1:
            minor_step = 1
        major_step = (
            major_grid_step if major_grid_step is not None
            else 5 * minor_step
        )
        minor_xticks = np.arange(x_min, x_max + 1, minor_step)
        minor_yticks = np.arange(y_min, y_max + 1, minor_step)
        major_xticks = np.arange(x_min, x_max + 1, major_step)
        major_yticks = np.arange(y_min, y_max + 1, major_step)
        ax.set_xticks(major_xticks)
        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor=True)
        ax.grid(which='minor', linestyle=minor_grid_linestyle, color=grid_color, linewidth=.25)
        ax.grid(which='major', linestyle=major_grid_linestyle, color=grid_color, linewidth=.5)
    
    if title is not None:
        ax.set_title(title, color=text_color)
    
    if ax is None:
        plt.show()


""" SKImage
"""


def get_sample_8_dir(subdir=""):
    return f"../tmp/preproc_images/sample_8/{subdir}"


def get_sample_100_dir(subdir=""):
    return f"../tmp/preproc_images/sample_100/{subdir}"


def get_all_1050_dir(subdir=""):
    return f"../tmp/preproc_images/all_1050/{subdir}"


def load_products_sample(images_dir):
    filenames, ids = get_file_names_and_ids(images_dir)
    rgb_imxs = for_all(load_image, filenames, const_args=images_dir)
    return rgb_imxs, ids, filenames


def load_the_sun(subdir="0_src/"):
    src_dir = get_sample_8_dir(subdir)
    filename = "216c6c3527984be3d7ad9023d5cd9bd1.jpg"
    return load_image(filename, src_dir, meth='pil')


def load_the_watch(subdir="0_src/"):
    src_dir = get_sample_8_dir(subdir)
    filename = "c705a5735a94aeee547d1798e3e46ec4.jpg"
    return load_image(filename, src_dir, meth='pil')


def load_the_pipe(subdir="0_src/"):
    src_dir = get_sample_8_dir(subdir)
    filename = "29def171d7e31d48571a52f0fb3e6b07.jpg"
    return load_image(filename, src_dir, meth='pil')


def load_the_pillow(subdir="0_src/"):
    src_dir = get_sample_8_dir(subdir)
    filename = "2eb07dc77e2fc7e3668fd7ed9b864039.jpg"
    return load_image(filename, src_dir, meth='pil')


def pipeline_step(
        locate_dir_f,
        transform_f=None,
        extract_f=None,
        input_subdir=None,
        output_subdir=None,
        input_cache=None,
        output_state=None,
        block_slice=None
):    
    start_time = get_start_time()

    if input_subdir is None and input_cache is None:
        print("pipeline_step ERROR")
    
    input_imxs, ids, filenames = None, None, None
    if input_cache is None:
        input_dir = locate_dir_f(input_subdir)
        input_imxs, ids, filenames = load_products_sample(input_dir)
    else:
        input_imxs, ids, filenames = input_cache

    if block_slice is not None:
        input_imxs = input_imxs[block_slice]
        ids = ids[block_slice]
        filenames = filenames[block_slice]

    if transform_f is None:
        transform_f = lambda x: x

    if output_state is None:
        output_state = "transformed"

    extracted, output_imxs = None, None
    if extract_f is None:
        output_imxs = transform_f(input_imxs)
    else:
        extracted = extract_f(input_imxs)
        output_imxs = transform_f(input_imxs, *extracted)

    output_dir = locate_dir_f(output_subdir)
    if output_subdir is not None:
        save_all(output_imxs, filenames, output_dir)

    print_time_perf(
        what=f"{len(ids)} images {output_state} {block_slice}",
        where=f"to {output_dir}" if output_subdir else f"from {input_dir}",
        start_time=start_time
    )

    if extracted is None:
        return output_imxs, ids, filenames
    else:
        return *extracted, output_imxs, ids, filenames



""" RGB to HSV (SKI 3.2)
"""


def _rgb_to_hsv_ski(rgb_imx):
    return ski.color.rgb2hsv(img_to_imx(rgb_imx))


def _rgb_to_hsv_pil(rgb_imx):
    return imx_to_img(rgb_imx).convert('HSV')


def rgb_to_hsv(rgb_imx, meth='pil'):
    if meth == 'pil':
        return img_to_imx(_rgb_to_hsv_pil(rgb_imx)) / 255
    if meth == 'ski':
        return _rgb_to_hsv_ski(rgb_imx)


def imshow_rgb_to_hsv(rgb_imx, hsv_imx, ax=None):
    fig = None
    if ax is None:
        fig, axes = plt.subplots(ncols=4, figsize=(10, 2))
        ax = np.ravel(axes)
    ax0, ax1, ax2, ax3 = ax

    hue_imx = hsv_imx[..., 0]
    sat_imx = hsv_imx[..., 1]
    val_imx = hsv_imx[..., 2]
    
    ax0.imshow(rgb_imx)
    ax0.set_title("RGB image")
    ax0.axis('off')

    ax1.imshow(hue_imx, cmap='hsv')
    ax1.set_title("Hue channel")
    ax1.axis('off')

    ax2.imshow(sat_imx)
    ax2.set_title("Saturation channel")
    ax2.axis('off')

    ax3.imshow(val_imx)
    ax3.set_title("Value channel")
    ax3.axis('off')

    if fig is not None:
        fig.tight_layout()
        plt.show()


def show_thres_img_hist(imx, threshold, channel_name, ax=None):
    binary_imx = imx > threshold

    fig = None
    if ax is None:
        fig, axes = plt.subplots(ncols=2, figsize=(10, 2))
        ax = np.ravel(axes)
    ax0, ax1 = ax

    ax0.hist(imx.ravel(), 512)
    ax0.set_title(
        f"Histogram of the {channel_name} "
        f"thresholded ({round(threshold, 2)}) image"
    )
    ax0.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)
    ax0.set_xbound(0.01, 0.99)
    ax0.set_yscale("log")

    ax1.imshow(binary_imx)
    ax1.set_title("Thresholded image")
    ax1.axis('off')

    if fig is not None:
        fig.tight_layout()
        plt.show()


""" Edges and lines (# SKI 4)
"""


""" SKI 4.3. Canny edge detector
"""


def get_canny_edge_map(gray_imx, sigma=5):
    return skif.canny(
        gray_imx,               # Input grayscale image
        sigma=sigma,            # Optional standard deviation of the Gaussian filter.
        # low_threshold=.1,     # Optional Lower bound for hysteresis thresholding (linking edges).
                                # If None, low_threshold is set to 10% of dtype’s max.
        # high_threshold=.2,    # Optional Upper bound for hysteresis thresholding (linking edges).
                                # If None, high_threshold is set to 20% of dtype’s max.
        # mask=mask,            # Optional Mask to limit the application of Canny to a certain area.
        # use_quantiles=True,   # If True then treat low_threshold and high_threshold as quantiles
                                # of the edge magnitude image, rather than absolute edge magnitude values.
                                # If True then the thresholds must be in the range [0, 1].
        # mode='wrap',          # Optional {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
                                # The mode parameter determines how the array borders
                                # are handled during Gaussian filtering, where cval is
                                # the value when mode is equal to ‘constant’.
        # cval=.5,              # Optional Value to fill past edges of input if `mode` is ‘constant’.
    )


def imshow_canny_edge_maps(gray_imx, edge_maps, sigmas, ax=None):
    fig = None
    if ax is None:
        fig, axes = plt.subplots(ncols=len(edge_maps)+1, figsize=(10, 2))
        ax = np.ravel(axes)

    ax[0].imshow(gray_imx, cmap='gray')
    ax[0].set_title('original image', fontsize=10)

    for i, (edge_map, sigma) in enumerate(zip(edge_maps, sigmas), start=1):
        ax[i].imshow(edge_map, cmap='gray')
        ax[i].set_title(fr'Canny filter, $\sigma={sigma}$', fontsize=10)

    for a in ax:
        a.axis('off')

    if fig is not None:
        fig.tight_layout()
        plt.show() 


def get_canny_edge_maps(gray_imx, sigmas):
    return for_all(
        get_canny_edge_map,
        const_kwargs={'gray_imx': gray_imx},
        kwargs_vect=[{'sigma': sigma} for sigma in sigmas]
    )


def get_all_canny_edge_maps(gray_imxs, sigmas):
    return for_all(
        get_canny_edge_maps,
        args_vect=gray_imxs,
        const_args=sigmas
    )


def for_all_imshow_cannys(gray_imxs, edge_maps_s, sigmas):
    nrows = len(gray_imxs)
    ncols = len(sigmas) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5 * ncols, 2 * nrows))
    for_all(
        imshow_canny_edge_maps,
        kwargs_vect=[
            {
                'gray_imx': gray_imx,
                'edge_maps': edge_imx,
                'ax': ax
            } for gray_imx, edge_imx, ax
            in zip(gray_imxs, edge_maps_s, axes)
        ],
        const_kwargs={'sigmas': sigmas}
    )
    fig.tight_layout()
    plt.show()


""" SKI 4.13. Edge operators
"""

def edge_filter(gray_imx, meth='sobel'):
    methods = {
        'sobel': filters.sobel,
        'roberts': filters.roberts,
        'scharr': filters.scharr,
        'prewitt': filters.prewitt,
        'farid': filters.farid,
    }
    filter = methods.get(meth, filters.sobel)
    return filter(gray_imx)


def imshow_edge_detection(gray_imx, edge_maps, edge_meths, ax=None):
    fig = None
    if ax is None:
        ncols = len(edge_maps)+1
        fig, axes = plt.subplots(ncols=ncols, figsize=(2 * ncols, 2))
        ax = np.ravel(axes)

    ax[0].imshow(gray_imx, cmap='gray')
    ax[0].set_title('original image', fontsize=10)

    for i, (edge_map, edge_meth) in enumerate(zip(edge_maps, edge_meths), start=1):
        ax[i].imshow(edge_map, cmap='gray')
        ax[i].set_title(f'{edge_meth} filter', fontsize=10)

    for a in ax:
        a.axis('off')

    if fig is not None:
        fig.tight_layout()
        plt.show() 


def all_edge_meths(gray_imx, edge_meths):
    return for_all(
        edge_filter,
        kwargs_vect=[{'meth': meth} for meth in edge_meths],
        const_kwargs={'gray_imx': gray_imx}   
    )


def get_all_edge_maps(gray_imxs, edge_meths):
    return for_all(
        all_edge_meths,
        args_vect=gray_imxs,
        const_args=edge_meths
    )


def for_all_imshow_all_edge_maps(gray_imxs, edge_maps_s, edge_meths):
    nrows = len(gray_imxs)
    ncols = len(edge_meths) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5 * ncols, 2 * nrows))
    for_all(
        imshow_edge_detection,
        kwargs_vect=[
            {
                'gray_imx': gray_imx,
                'edge_maps': edge_map,
                'ax': ax
            } for gray_imx, edge_map, ax
            in zip(gray_imxs, edge_maps_s, axes)
        ],
        const_kwargs={'edge_meths': edge_meths}
    )
    fig.tight_layout()
    plt.show()


""" Detection of features and objects (# SKI 8)
"""


""" SKI 8.1. DAISY
"""

""" Return a grid of DAISY descriptors for the given image as an array dimensionality (P, Q, R) where
* P = ceil((M - radius*2) / step)
* Q = ceil((N - radius*2) / step)
* R = (rings * histograms + 1) * orientations
"""
def get_daisy_desc(gray_imx, n_desc=9):
    h, w = gray_imx.shape
    step = math.floor(math.sqrt(h * w / n_desc))
    # step=4, radius=15, rings=3, histograms=8, orientations=8,
    # normalization='l1', sigmas=None, ring_radii=None, visualize=False
    return skif.daisy(
        gray_imx,              # Input grayscale image
        step=step,             # Optional distance between descriptor sampling points.
        radius=step//3,        # Optional radius (in pixels) of the outermost ring.
        rings=2,               # Optional number of rings.
        histograms=6,          # Optional number of histograms sampled per ring.
        # orientations=8,      # Optional number of orientations (bins) per histogram.
        # How to normalize the descriptors
        #        'l1': L1-normalization of each descriptor.
        #        'l2': L2-normalization of each descriptor.
        #        'daisy': L2-normalization of individual histograms.
        #        'off': Disable normalization.
        # normalization='daisy',
        # sigmas=sigmas,       # 1D array of float, optional
        # ring_radii=radii,    # 1D array of int, optional
        visualize=True,        # Optional : generate a visualization of the DAISY descriptors
    )


def daisy_all(ims, n_desc=9):
    return for_all(get_daisy_desc, ims, const_kwargs={'n_desc': n_desc})


""" Commons
"""


def draw_keypoints(gray_imx, keypoints, scales=None):
    im = pil.fromarray(gray_imx)
    im = im.convert('RGB')
    draw_obj = draw.Draw(im)
    if scales is None:
        scales = [5] * len(keypoints)
    for keypoint, scale in zip(keypoints, scales):
        y, x = keypoint
        r = 2 ** (1 + scale / 2)
        draw_obj.ellipse((x - r, y - r, x + r, y + r), outline='red', width=3)
    return np.array(im)




""" SKI 8.5. CENSURE
"""

def get_censure_desc(gray_imx):
    censure = skif.CENSURE(
        # min_scale=1,
        # max_scale=7,
        # mode='DoB',
        # non_max_threshold=0.15,
        # line_threshold=10
    )
    censure.detect(gray_imx)
    return censure.keypoints, censure.scales


def censure_all(gray_imxs):
    return for_all(get_censure_desc, gray_imxs) #, const_kwargs={'min_scale': ...})


def get_censured_imxs(gray_imxs, keypoints, scales):
    return for_all(draw_keypoints, zip(gray_imxs, keypoints, scales))


""" SKI 8.10. ORB
"""


def get_orb_desc(gray_imx, id=None):
    if id is not None:
        print("get_orb_desc :", id)
    orb = skif.ORB(
        # downscale=1.2,
        # n_scales=8,
        # n_keypoints=500,
        # fast_n=9,
        # fast_threshold=0.08,
        # harris_k=0.04
    )
    #orb.detect_and_extract(gray_imx)
    #return orb.keypoints, orb.descriptors
    try:
        orb.detect_and_extract(gray_imx)
    except RuntimeError:
        print(f"Warning: ORB failed to find features in image {id}")
        print("Retry after improving image contrast between adjacent pixels.")
        # If SIFT fails to find any features, try to improve contrast and retry
        p2, p98 = np.percentile(gray_imx, (2, 98))
        gray_imx = exposure.rescale_intensity(gray_imx, in_range=(p2, p98))
        try:
            orb.detect_and_extract(gray_imx)
        except RuntimeError:
            # If SIFT fails again, print a warning and return None values
            print(f"Warning: ORB definitively failed to find features in image {id}")
            return None, None
    return orb.keypoints, orb.descriptors


def orb_all(gray_imxs, ids=None):
    if ids is None:
        return for_all(get_orb_desc, gray_imxs) #, const_kwargs={'descriptor_size': ...})
    else:
        return for_all(get_orb_desc, zip(gray_imxs, ids))


def get_orbed_imxs(gray_imxs, keypoints):
    return for_all(draw_keypoints, zip(gray_imxs, keypoints))


""" SKI 8.12. BRIEF
"""


def get_brief_desc(gray_imx, return_keypoints=True):
    brief = skif.BRIEF(
        # descriptor_size=256,
        # patch_size=49,
        # mode='normal',
        # sigma=1,
        # sample_seed=1
    )
    keypoints = skif.corner_peaks(
        skif.corner_harris(gray_imx),
        min_distance=5,
        threshold_rel=0.1
    )

    # brief ne conserve pas trace des keypoints conservés vs. discardés
    # cela oblige à itérer
    if not return_keypoints:
        brief.extract(gray_imx, keypoints)
        return brief.descriptors

    kept_keypoints = []
    descriptors = []
    for keypoint in keypoints:
        #print("keypoint:", keypoint)
        brief.extract(gray_imx, np.array([keypoint]))
        if len(brief.descriptors) > 0:
            descriptors.extend(brief.descriptors)
            kept_keypoints.append(keypoint)

    return np.array(kept_keypoints), np.array(descriptors)


def brief_all(gray_imxs, return_keypoints=True):
    return for_all(
        get_brief_desc,
        gray_imxs,
        const_args=return_keypoints
    ) #, const_kwargs={'descriptor_size': ...})


def get_briefed_imxs(gray_imxs, keypoints):
    return for_all(draw_keypoints, zip(gray_imxs, keypoints))


""" SKI 8.13. SIFT
"""


def get_sift_desc(gray_imx, id=None):
    if id is not None:
        print("get_sift_desc :", id)
    sift = skif.SIFT(
        # upsampling=2,
        # n_octaves=8,
        # n_scales=3,
        # sigma_min=1.6,
        # sigma_in=0.5,
        # c_dog=0.013333333333333334,
        # c_edge=10,
        # n_bins=36,
        # lambda_ori=1.5,
        # c_max=0.8,
        # lambda_descr=6,
        # n_hist=4,
        # n_ori=8
    )
    #sift.detect_and_extract(gray_imx)
    #return sift.keypoints, sift.descriptors
    try:
        sift.detect_and_extract(gray_imx)
    except RuntimeError:
        print(f"Warning: SIFT failed to find features in image {id}")
        print("Retry after improving image contrast between adjacent pixels.")
        # If SIFT fails to find any features, try to improve contrast and retry
        p2, p98 = np.percentile(gray_imx, (2, 98))
        gray_imx = exposure.rescale_intensity(gray_imx, in_range=(p2, p98))
        try:
            sift.detect_and_extract(gray_imx)
        except RuntimeError:
            # If SIFT fails again, print a warning and return None values
            print(f"Warning: SIFT definitively failed to find features in image {id}")
            return None, None
    return sift.keypoints, sift.descriptors


def sift_all(gray_imxs, ids=None):
    if ids is None:
        return for_all(get_sift_desc, gray_imxs) #, const_kwargs={'descriptor_size': ...})
    else:
        return for_all(get_sift_desc, zip(gray_imxs, ids))


def get_sifted_imxs(gray_imxs, keypoints):
    return for_all(draw_keypoints, zip(gray_imxs, keypoints))


""" OpenCV
"""


def get_ocv_kpts_data(kpts, property_names=None, as_array=False):
    if kpts is None or len(kpts) == 0:
        return None
    if property_names is None:
        members = dir(kpts[0])
        property_names = [m for m in members if not m.startswith("__")]
    if as_array:
        return np.vstack([
            vars_without__dict__(kpt, property_names, as_array)
            for kpt in kpts
        ])
    else:
        return [
            vars_without__dict__(kpt, property_names, as_array)
            for kpt in kpts
        ]


def ocv_ims_kpts_to_array(ocv_ims_kpts):
    return [
        get_ocv_kpts_data(kpt, ['pt'], as_array=True)
        for kpt in ocv_ims_kpts
    ]


""" OCV Harris corner
"""

# blockSize, ksize, k
def get_harris_corners(gray_imx, block_size=2, ksize=3, k=.04):
    float_gray_imx = np.float32(gray_imx)

    # Extract corners
    corners = cv.cornerHarris(
        # Input 8-bit or floating-point 32-bit, single-channel image. 
        float_gray_imx,
        # blockSize Neighborhood size
        blockSize=block_size,
        # Aperture parameter for the Sobel operator.  
        ksize=ksize,
        # Harris detector free parameter in $R = \det{M} - k(\tr{M})^2$.
        k=k
    )
    
    # Result is dilated for marking the corners
    corners = cv.dilate(corners, kernel=None)

    return corners


def refine_harris_corners(gray_imx, centroids):
    # define the criteria to stop and refine the corners
    # on peut difficilement faire plus contre-intuitif et mal documenté
    # https://stackoverflow.com/questions/70751242/difference-of-termcriteria-type-in-opencv-count-and-max-iter
    criteria = (
        cv.TERM_CRITERIA_EPS
        + cv.TERM_CRITERIA_MAX_ITER,    # Flag defining termination condition
        100,                            # Iteration max
        0.001                           # Precision
    )

    corners = cv.cornerSubPix(
        # Input single-channel, 8-bit or float image.
        gray_imx,
        # Initial coordinates of the input corners and refined coordinates provided for output.
        # The documentation doesn't say that float32 is required when the output
        # of connectedComponentsWithStats is float64!!
        corners=np.float32(centroids),  
        # Half of the side length of the search window.
        # For example, if winSize=Size(5,5), then a (5∗2+1)×(5∗2+1)=11×11 search window is used. 
        winSize=(5, 5),
        # Half of the size of the dead region in the middle of the search zone over which
        # the summation in the formula below is not done.
        # It is used sometimes to avoid possible singularities of the autocorrelation matrix.
        # The value of (-1,-1) indicates that there is no such a size. 
        zeroZone=(-1, -1),
        # Criteria for termination of the iterative process of corner refinement.
        # That is, the process of corner position refinement stops either after criteria.maxCount
        # iterations or when the corner position moves by less than criteria.epsilon on some iteration.
        criteria=criteria
    )

    return corners


def get_harris_corners_data(
    gray_imx,
    block_size=2,
    ksize=3,
    k=.04,
    thresh_pct=.01
):
    corners = get_harris_corners(
        gray_imx,
        block_size=block_size,
        ksize=ksize,
        k=k
    )

    thresh, corners = cv.threshold(
        corners,
        thresh=thresh_pct*corners.max(),
        maxval=255,
        type=cv.THRESH_BINARY
    )

    corners = np.uint8(corners)
    
    (
        n_labels,     # count of labels
        labels,       # labeled image
        stats,        # ndarray [labels] x [x, y, w, h, area]
        centroids     # label centers
    ) = cv.connectedComponentsWithStats(corners)

    corners = refine_harris_corners(gray_imx, centroids)

    np_data = np.hstack([centroids, corners, stats])
    data = pd.DataFrame(np_data)
    data.columns = pd.MultiIndex.from_tuples([
        ('centroid', 'x'), ('centroid', 'y'),
        ('corner', 'x'), ('corner', 'y'),
        ('box', 'x'), ('box', 'y'), ('box', 'w'), ('box', 'h'), ('box', 'area') 
    ])
    data.index.name = 'label_id'

    return data


def draw_harris_corners(gray_imx=None, corners=None, centroids=None, s=3):
    if gray_imx is None and corners is None and centroids is None:
        return None
    if corners is None and centroids is None:
        return gray_imx

    imx = None
    # Create a new image from scratch if no input image is provided
    if gray_imx is None:
        # The center of original image is the first line corners or centroids
        w_h = None
        if centroids is None:
            w_h = np.uint0(np.around(corners[0] * 2))
        else:
            w_h = np.uint0(np.around(centroids[0] * 2))
        imx = np.zeros((w_h[1], w_h[0], 3), dtype=np.uint8)
    else:
        """imx = pil.fromarray(gray_imx)
        imx = imx.convert('RGB')
        imx = np.array(imx)
        more simple and direct :
        """
        imx = np.stack([gray_imx, gray_imx, gray_imx], axis=-1)

    if centroids is not None:
        centroids = np.int0(centroids)
        for x, y in centroids:
            cv.rectangle(imx, (x-s, y-s), (x+s, y+s), (0, 255, 0), -1)

    if corners is not None:
        corners = np.int0(corners)
        for x, y in corners:
            cv.rectangle(imx, (x-s, y-s), (x+s, y+s), (255, 0, 0), -1)

    return imx


def ocv_harris_all(gray_imxs):
    return for_all(get_harris_corners_data, gray_imxs) #, const_kwargs={'descriptor_size': ...})


def get_harrised_imxs(gray_imxs, corners, centroids):
    return for_all(draw_harris_corners, zip(gray_imxs, corners, centroids))


""" OCV Shi-Tomasi corner
"""


# max_corners=50, quality_level=.01, min_distance=10,  block_size=3
def get_shi_tomasi_corners(
    gray_imx,
    max_corners=50,
    quality_level=.01,
    min_distance=10,
    block_size=3,
):
    float_gray_imx = np.float32(gray_imx)

    # Extract corners
    corners = cv.goodFeaturesToTrack(
        # Input 8-bit or floating-point 32-bit, single-channel image. 
        float_gray_imx,
        # Maximum number of corners to return. If there are more corners than are found,
        # the strongest of them is returned. `maxCorners <= 0` implies that no limit
        # on the maximum is set and all detected corners are returned.
        maxCorners=max_corners,
        # Parameter characterizing the minimal accepted quality of image corners.
        # The parameter value is multiplied by the best corner quality measure,
        # which is the minimal eigenvalue (see #cornerMinEigenVal ) or
        # the Harris function response (see #cornerHarris ).
        # The corners with the quality measure less than the product are rejected.
        # For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01,
        # then all the corners with the quality measure less than 15 are rejected.
        qualityLevel=quality_level,
        # Harris detector free parameter in $R = \det{M} - k(\tr{M})^2$
        # Minimum possible Euclidean distance between the returned corners.
        minDistance=min_distance,
        # Size of an average block for computing a derivative covariation matrix
        # over each pixel neighborhood. See cornerEigenValsAndVecs.
        blockSize=block_size
        # Parameter indicating whether to use a Harris detector (see cornerHarris) or cornerMinEigenVal. 
        # useHarrisDetector=True,
        # Free parameter of the Harris detector.
        # k=.04
    )
    # the values are already integers, but as floats,
    # and the second dimension is unnecessary 
    return np.int0(corners[:, 0, :])


def draw_shi_tomasi_corners(gray_imx, corners, radius=10):
    rgb_imx = np.stack([gray_imx, gray_imx, gray_imx], axis=-1)
    for x, y in corners:
        cv.circle(rgb_imx, center=(x, y), radius=radius, color=255, thickness=-1)
    return rgb_imx


def ocv_shi_tomasi_all(gray_imxs):
    return for_all(get_shi_tomasi_corners, gray_imxs) #, const_kwargs={'descriptor_size': ...})


def get_shi_tomasied_imxs(gray_imxs, corners):
    return for_all(draw_shi_tomasi_corners, zip(gray_imxs, corners))



""" OCV FAST
"""

def get_fast_kpts(gray_imx):
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create(
        # threshold=10,
        # nonmaxSuppression=True
    )
    # Find the keypoints
    return np.array(fast.detect(image=gray_imx, mask=None))


def ocv_fast_all(gray_imxs):
    return for_all(get_fast_kpts, gray_imxs) #, const_kwargs={'descriptor_size': ...})


def get_fasted_imxs(gray_imxs, keypoints):
    return for_all(
        cv.drawKeypoints,
        zip(gray_imxs, keypoints),
        const_kwargs={'outImage': None, 'color': (255, 0, 0)}
    )

#cv.drawKeypoints(imx, kpts, outImage=None, color=(255, 0, 0))

# Avec ce qui suit, je pourrais apporter une réponse complémentaire à :
# https://stackoverflow.com/questions/47623014/converting-a-list-of-objects-to-a-pandas-dataframe
def vars_without__dict__(kpt, property_names=None, as_array=False):
    if property_names is None:
        members = dir(kpt)
        property_names = [m for m in members if not m.startswith("__")]
    if as_array:
        return np.array([
            getattr(kpt, prop)  # accès à la valeur de l'attribut de nom prop
            for prop in property_names
        ], dtype=object)
    else:
        return {
            prop: getattr(kpt, prop)  # accès à la valeur de l'attribut de nom prop
            for prop in property_names
        }


""" OCV SIFT
"""


# Most recent readable API^^ doc for OpenCV : https://docs.opencv.org/3.0-beta/
def get_ocv_sift_kpts_descs(gray_imx):
    # Initiate SIFT object with default values
    sift = cv.SIFT_create(
        # nfeatures = 0
        #   The number of best features to retain. The features are ranked by their scores
        #   (measured in SIFT algorithm as the local contrast)

        # nOctaveLayers = 3
        #   The number of layers in each octave. 3 is the value used in D. Lowe paper.
        #   The number of octaves is computed automatically from the image resolution.

        # contrastThreshold = .04
        #   The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
        #   The larger the threshold, the less features are produced by the detector.
        # note The contrast threshold will be divided by nOctaveLayers when the filtering is applied.
        # When nOctaveLayers is set to default and if you want to use the value used in D.
        # Lowe paper, 0.03, set this argument to 0.09.

        # edgeThreshold = 10
        #   The threshold used to filter out edge-like features.
        #   Note that the its meaning is different from the contrastThreshold,
        #   i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).

        # sigma = 1.6
        #   The sigma of the Gaussian applied to the input image at the octave #0.
        #   If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    )

    # Find the keypoints
    kpts = sift.detect(gray_imx, mask=None)

    # Compute descriptors from keypoints
    return sift.compute(gray_imx, keypoints=kpts)


def ocv_sift_all(gray_imxs):
    return for_all(get_ocv_sift_kpts_descs, gray_imxs) #, const_kwargs={'descriptor_size': ...})


def get_ocv_marked_imxs(gray_imxs, keypoints):
    return for_all(
        cv.drawKeypoints,
        zip(gray_imxs, keypoints),
        const_kwargs={
            'outImage': None,
            'flags': cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        } #, 'color': (255, 0, 0)}
    )



""" OCV BRIEF
"""

def get_ocv_brief_kpts_descs(gray_imx):
    # Initiate FAST detector
    star = cv.xfeatures2d.StarDetector_create()

    # Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create(
        bytes=64,              # legth of the descriptor in bytes, valid values are: 16, 32 (default) or 64 . 
        use_orientation=True   # sample patterns using keypoints orientation, disabled by default. 
    )

    # find the keypoints with STAR
    kpts = star.detect(gray_imx, mask=None)

    # compute the descriptors with BRIEF
    kpts, descs = brief.compute(gray_imx, kpts)

    return kpts, descs


def ocv_brief_all(gray_imxs):
    return for_all(get_ocv_brief_kpts_descs, gray_imxs) #, const_kwargs={'descriptor_size': ...})


"""OCV ORB
"""

def get_ocv_orb_kpts_descs(gray_imx):
    # Initiate ORB detector
    orb = cv.ORB_create(
        # nfeatures
        #   The maximum number of features to retain.
        # scaleFactor
        #   Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid,
        #   where each next level has 4x less pixels than the previous, but such a big scale factor will
        #   degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
        #    will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.
        # nlevels
        #   The number of pyramid levels. The smallest level will have linear size equal
        #   to input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
        # edgeThreshold
        #   This is size of the border where the features are not detected.
        #   It should roughly match the patchSize parameter.
        # firstLevel
        #   The level of pyramid to put source image to.
        #   Previous layers are filled with upscaled source image.
        # WTA_K
        #   The number of points that produce each element of the oriented BRIEF descriptor.
        #   The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
        #   so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
        #   random points (of course, those point coordinates are random, but they are generated from the
        #   pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
        #   rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2).
        #   Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
        #   denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
        #   bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
        # scoreType
        #   The default HARRIS_SCORE means that Harris algorithm is used to rank features
        #   (the score is written to KeyPoint::score and is used to retain best nfeatures features);
        #   FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
        #   but it is a little faster to compute.
        # patchSize
        #   size of the patch used by the oriented BRIEF descriptor.
        #   Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.
        # fastThreshold the fast threshold
    )

    # find the keypoints with ORB
    kpts = orb.detect(gray_imx, None)

    # compute the descriptors with ORB
    kpts, descs = orb.compute(gray_imx, kpts)

    return kpts, descs


def ocv_orb_all(gray_imxs):
    return for_all(get_ocv_orb_kpts_descs, gray_imxs) #, const_kwargs={'descriptor_size': ...})


""" Data assembly and persistency
"""

## KO : NE FONCTIONNE PAS
def image_desc_dtypes_v2(keypoints, descriptors=None):
    # deprecated
    # determine dtypes for each column if not specified
    # to extract only dtypes : np.dtype(image_desc_dtypes(keypoints, descriptors))
    dtypes = []
    dtypes.append(("id", "O"))
    for i in range(keypoints.shape[1]):
        dtype = str(keypoints.dtype)
        dtypes.append((f"kp_{i}", dtype))
    if descriptors is not None:
        for i in range(descriptors.shape[1]):
            dtype = str(descriptors.dtype)
            dtypes.append((f"desc_{i}", dtype))
    return dtypes

## KO : NE FONCTIONNE PAS
def image_desc_hstack_v2(id, keypoints, descriptors=None, dtypes=None):
    # deprecated
    h = keypoints.shape[0]
    if h == 0:
        return None
    if dtypes is None:
        dtypes = image_desc_dtypes_v2(keypoints, descriptors)
    id_rep = np.full((h, 1), id) #, dtype=dtypes[0])
    #id_rep = np.empty((h, 1), dtype=dtypes[0])
    #id_rep[:, 0] = id
    if descriptors is None:
        # id_rep = id_rep.astype(dtypes[0])
        #keypoints = keypoints.astype(dtypes[1:])
        print(np.dtype(dtypes))
        desc_hstack = np.hstack([id_rep, keypoints])
        desc_hstack = desc_hstack.astype(dtypes)
        print(desc_hstack.dtype)
        return desc_hstack
    else:
        # id_rep = id_rep.astype(dtypes[0])
        #keypoints = keypoints.astype(dtypes[1:keypoints.shape[1]+1])
        #descriptors = descriptors.astype(dtypes[keypoints.shape[1]+1:])
        print(np.dtype(dtypes))
        desc_hstack = np.hstack([id_rep, keypoints, descriptors])
        display(desc_hstack[0])
        desc_hstack = desc_hstack.astype(dtypes)
        print(desc_hstack.dtype)
        return desc_hstack


# réunion des tableaux en un seul,
# avec index de l'image en tête,
# puis coordonnées et descripteur des points d'intérêt et descripteur
def image_desc_hstack_v1(id, keypoints, descriptors=None):
    # deprecated
    h = keypoints.shape[0]
    if h == 0:
        return None
    id_rep = np.full((h, 1), id)
    if descriptors is None:
        return np.hstack([id_rep, keypoints])
    else:
        return np.hstack([id_rep, keypoints, descriptors])


def images_descriptions_array_v1(ids, keypoints, descriptors=None):
    # deprecated
    if descriptors is None:
        return np.vstack([
            image_desc_hstack_v1(i, k)
            for i, k in zip(ids, keypoints)
            if k is not None and k.shape[0] > 0
        ])
    else:
        return np.vstack([
            image_desc_hstack_v1(i, k, d)
            for i, k, d in zip(ids, keypoints, descriptors)
            if k is not None and k.shape[0] > 0
        ])


def images_descriptions_data_v1(ids, keypoints, descriptors=None):
    # deprecated : use images_descriptions_data instead
    descs_array = images_descriptions_array_v1(ids, keypoints, descriptors)
    # just never redo this on a big one : pd.DataFrame.from_records(descs_array)
    data = pd.DataFrame(descs_array)
    cols = ['id', 'y', 'x']
    if descriptors is not None:
        desc_len = descriptors[0].shape[1]
        cols += [str(i) for i in range(desc_len)]
    data.columns = cols
    data.set_index('id', inplace=True)
    data[data.columns[:2]] = data[data.columns[:2]].astype(keypoints.dtype)
    data[data.columns[2:]] = data[data.columns[2:]].astype(descriptors.dtype)
    return data


def get_id_rep_v1(ids, kpts):
    # deprecated : plus compliquée et moins performante et surtout shape (n,)
    return np.vstack([
        np.full((kpt.shape[0], 1), id)
        for id, kpt in zip(ids, kpts)
        if kpt is not None and kpt.shape[0] > 0
    ])


def get_id_rep(ids, kpts):
    return np.concatenate([
        np.repeat(id, kpt.shape[0])
        for id, kpt in zip(ids, kpts)
    ])


def images_descriptions_data(ids, kpts, descs=None):
    id_rep = get_id_rep(ids, kpts)
    kpts_stacked = np.vstack([kpt for kpt in kpts if kpt is not None])
    kpts_data = pd.DataFrame(kpts_stacked, index=id_rep).astype(dtype=int)
    kpts_data.index.names = ['id']
    kpts_data.columns = ['y', 'x']
    if descs is None:
        return kpts_data
    else:
        descs_stacked = np.vstack([desc for desc in descs if desc is not None])
        descs_data = pd.DataFrame(descs_stacked, index=id_rep)
        descs_data.index.names = ['id']
        descs_data.columns = [str(i) for i in range(descs[0].shape[1])]
        return pd.concat([kpts_data, descs_data], axis=1)


def save_desc_data(data, data_name, root_dir=None):
    if root_dir is not None:
        create_if_not_exist(root_dir)
    file_path = f"{root_dir}{data_name}.parquet"
    data.to_parquet(file_path)
    file_size = get_file_size(file_path)
    print(f"{data_name} {data.shape}")
    print(*format_iB(file_size), "parquet file saved") 


def load_desc_data(
        dataname: str,
        root_dir: Optional[str] = None
) -> pd.DataFrame:
    r"""Load the data with given dataname from the specified root directory.

    Parameters
    ----------
    dataname : str
        The name of the data to load (without file extension).
    root_dir : str, optional
        The path to the directory where the data is stored.
        If None, the current working directory is used.

    Returns
    -------
    pd.DataFrame
        The loaded data as a pandas DataFrame.
    """
    return pd.read_parquet(f"{root_dir}{dataname}.parquet")


""" Other data extraction
"""


""" Images metadata
"""

def load_images_and_ids(images_dir):
    filenames, ids = get_file_names_and_ids(images_dir)
    if len(filenames) == 0:
        print(f"Empty folder case ({images_dir}) : pass")
        return None, None
    if len(filenames) == 1:
        print(f"NPZ case ({images_dir}) : for now, pass")
        return None, None
    imxs_imgs_zip = for_all(_load_image_pil_open, filenames, const_args=images_dir)
    imgs = imxs_imgs_zip[1]
    return imgs, ids


def get_images_info_data(imgs, ids):
    records = np.array([
        (id, img.format, img.mode, img.size[0], img.size[1])
        for id, img in zip(ids, imgs)
    ])
    data = pd.DataFrame.from_records(records)
    data.columns = ['id', 'format', 'mode', 'width', 'height']
    data.set_index('id', inplace=True)
    data.width = data.width.astype(int, copy=False)
    data.height = data.height.astype(int, copy=False)
    data['size_px'] = data.width * data.height
    data['size_kb'] = (data.size_px / 8 / 1_024).astype(int)
    return data


def subdirs(root_dir):
    pathes = glob.glob(f"**/", root_dir=root_dir, recursive=True)
    return [path.replace("\\", "/") for path in pathes]


def extract_info_data(root_dir, images_path):
    images_dir = root_dir + images_path
    info_data_filename = images_path.replace("/", "_").replace(" ", "_") + "format.csv"
    info_data_path = root_dir + info_data_filename
    print(info_data_filename)
    imgs, ids = load_images_and_ids(images_dir)
    if imgs is None or len(imgs) == 0:
        return
    data = get_images_info_data(imgs, ids)
    data.to_csv(info_data_path, encoding='utf8')




import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

import os
from utils.caching import ext_cache, get_hash

from typing import Iterable, Tuple, List, Callable
from .similarities import find_most_interesting_words


def animate_k(out_filename : str, k_values : list, display_function : Callable,
              refresh : bool = False,
              *args, **kwargs):
    """Create video animation for displaying different values for k in the k-rank approximation of the terms-docs matrix.

    Parameters
    ----------
    out_filename : str
        output filename
    k_values : list
        values of k to animate
    display_function : Callable
        display function. C
    refresh : bool, optional
        if True, creates the animation again, regardless if the cache checkfile exists or not. By default False
    """
    cached = ext_cache(out_filename, k_values, display_function, *args, *list(kwargs.values()))
    if os.path.exists(out_filename) and cached and not refresh:
        return

    print("Creating animation")
    # Create a figure and axis
    w, h, dpi = 640, 640, 70
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)

    # Define the update function for the animation
    def update(k):
        print(f'k={k}', end='\r')
        ax.clear()
        display_function(*args, **kwargs, ax=ax, k=k)
        fig.tight_layout(pad=0.2)
        return ax.artists

    ani = FuncAnimation(fig, update, frames=k_values, blit=True)
    ani.save(out_filename, writer='ffmpeg', fps=1)
    print('Done'.ljust(10))
    fig.clf()


def plot_docs(vt : np.ndarray, s : np.ndarray, dimensions : Tuple[int] = (0,1), labels : Iterable[str] = None,
              subset : np.ndarray = None, normalize : bool = True, k : int = None,
              subsample_size : int = None,
              ax : matplotlib.axes.Axes = None, show_grid : bool = True, scatter_kw={},
              verbose=True):
    """Plot the docs onto the two specified dimensions of the LSA space.

    The matrix `vt` encodes the docs. Optionally, only a subset of the docs can be shown.

    Parameters
    ----------
    vh : np.ndarray
        2D array, obtained from the SVD of the terms-docs or tf-idf matrix.
        It's the SVD matrix related to the docs. It has dimensions `(d, n_docs)`, where `d=min(n_words,n_docs)`. Its columns
        correspond to the docs.
    s : np.ndarray
        1D array of the singular values, sorted in descending order.
    dimensions : Tuple[int], optional
        the 2 LSA dimensions onto which plot the docs vectors, by default `(0,1)`
    labels : Iterable[str], optional
        List of string labels to assign to each point in the plot.
        If not provided, no label is shown.
    subset : np.ndarray, optional
        Array of integers, containing the indicies of the docs to plot.
        If not provided, all docs are plotted.
    normalize : bool, optional
        Whether to normalize or not the doc vectors, by default True.
    k : int, optional
        k-rank approximation of the LSA, by default 100. Used only if `normalize` is True.
    subsample_size : int, optional
        use only a sample of this size in the scatter plot. Might be useful if there are a lot of documents to plot.
        If not provided, all points are used. Used only if `subset` is not passed.
    ax : matplotlib.axes.Axes, optional
        axes onto which to show the plot. If not provided, a new figure instance with axes is created.
    show_grid : bool
        by default True.
    **scatter_kw : dict
        keyword arguments passed to ax.scatter()
    """
    if subset is None:
        subset = list(range(vt.shape[1]))
    vt = vt[:,subset]

    vt_s = s[:,np.newaxis] * vt
    if normalize:
        norms = np.sqrt(np.sum(np.square(vt_s[:k]), axis=0))
        vt_s = vt_s[:k]/np.where(norms==0, 1, norms)

    np.random.seed(42)
    if subsample_size is not None:
        if verbose:
            print(f"Showing only {subsample_size} datapoints out of {vt_s.shape[1]}")
        subsample_indices = np.random.choice(vt_s.shape[1], subsample_size, replace=False)
        vt_s = np.take(vt_s, subsample_indices, axis=1)

    dim_x, dim_y = dimensions

    if ax is None:
        w, h, dpi = 640, 640, 50
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)

    ax.scatter(vt_s[dim_x], vt_s[dim_y], **scatter_kw)
    ax.scatter(0, 0, c='k', marker='s', label='origin')

    if labels is not None:
        for i, label in enumerate(labels[subset]):
            ax.annotate(label, (vt_s[dim_x,i], vt_s[dim_y,i]))

    title_str = 'Normalized ' if normalize else ''
    title_str += f'LSA doc vectors: dims ({dim_x}, {dim_y})'

    if normalize:
        title_str += f', k={k}'
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02,1.02)

    ax.set_title(title_str)
    ax.set_xlabel(f'LSA dim {dim_x}')
    ax.set_ylabel(f'LSA dim {dim_y}')
    ax.legend()

    if show_grid:
        ax.grid()

    return ax


def plot_words(u : np.ndarray, s : np.ndarray, dimensions : Tuple[int] = (0,1), labels : Iterable[int] = None,
               subset : np.ndarray = None, normalize : bool = True, k : int = 100,
               subsample_size : int = None,
               ax : matplotlib.axes.Axes = None, show_grid : bool = True, scatter_kw={}):
    """Plot the words onto the two specified dimensions of the LSA space.

    The matrix `u` encodes the docs. Optionally, only a subset of the docs can be shown,
    i.e. only a subset of the rows of u are considered.

    Parameters
    ----------
    u : np.ndarray
        Bidimensional array, obtained from the SVD of the terms-docs or tf-idf matrix.
        It's the SVD matrix related to the words. It has dimensions (n_words,d), where d=min{n_words,n_docs}. Its rows
        correspond to the words.
    s : np.ndarray
        1D array of the singular values, sorted in descending order.
    dimensions : Tuple[int], optional
        the 2 LSA dimensions onto which plot the word vectors, by default `(0,1)`
    labels : Iterable[str], optional
        List of string labels to assign to each point in the plot.
        If not provided, no label is shown.
    subset : np.ndarray, optional
        Array of integers, containing the indicies of the words to plot.
        If not provided, all words are plotted.
    normalize : bool, optional
        Whether to normalize or not the words vectors, by default True.
    k : int, optional
        k-rank approximation of the LSA, by default 100. Used only if `normalize` is True.
    subsample_size : int, optional
        use only a sample of this size in the scatter plot. Might be useful if there are a lot of documents to plot.
        If not provided, all points are used. Used only if `subset` is not passed.
    ax : matplotlib.axes.Axes, optional
        axes onto which to show the plot. If not provided, a new figure instance with axes is created.
    show_grid : bool
        by default True.
    **scatter_kw : dict
        keyword arguments passed to ax.scatter()
    """
    if subset is None:
        subset = list(range(u.shape[0]))
    u = u[subset,:]

    u_s = u * s[np.newaxis,:]
    if normalize:
        norms = np.sqrt(np.sum(np.square(u_s[:,:k]), axis=1))
        u_s = u_s[:,:k]/np.where(norms==0, 1, norms)[:,np.newaxis]

    if subsample_size is not None:
        print(f"Showing only {subsample_size} datapoints out of {u_s.shape[0]}")
        subsample_indices = np.random.choice(u_s.shape[0], subsample_size, replace=False)
        u_s = np.take(u_s, subsample_indices, axis=0)

    dim_x, dim_y = dimensions

    if ax is None:
        w, h, dpi = 640, 640, 50
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)

    ax.scatter(u_s[:,dim_x], u_s[:,dim_y], **scatter_kw)
    ax.scatter(0, 0, c='k', marker='s', label='origin')

    if labels is not None:
        for i, label in enumerate([labels[i] for i in subset]):
            ax.annotate(label, (u_s[i,dim_x], u_s[i,dim_y]))

    title_str = 'Normalized ' if normalize else ''
    title_str += f'LSA doc vectors: dims ({dim_x}, {dim_y})'

    if normalize:
        title_str += f', k={k}'
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02,1.02)

    ax.set_title(title_str)
    ax.set_xlabel(f'LSA dim {dim_x}')
    ax.set_ylabel(f'LSA dim {dim_y}')
    ax.legend()

    if show_grid:
        ax.grid()

    return ax


def plot_genres_analysis(vh : np.ndarray, s : np.ndarray, df : pd.DataFrame, genres : List[str], 
                         dimensions : Tuple[int] = (0,1), words : List[str] = None, plot_most_relevant_words : bool = False, 
                         n : int = 1, u : np.ndarray = None, voc : np.ndarray = None, normalize : bool = False, 
                         k : int = 100, delete_intersection : bool = True, ax=None, scatter_kw={}):
    """Make a plot for analyzing the given genres of interest.

    Basically, we plot the movies onto the two specified dimensions of the LSA space, coloring the points in different ways
    according to their genre.
    The given movies are encoded as column vectors in the matrix `vh`.

    Optionally, a list of words to be plotted can be specified. In this way, we can see the relation of these words with the
    genres of interest.

    Still optionally, we can ask to plot, for each genre, the `n` most relevant words. Namely, the `n` words which are the
    most similar to the movies with that genre.
    To be more specific, for each word, its average cosine similarity computed w.r.t. all the movies with that genre is
    calculated, and then the `n` words with biggest score are taken.
    In the plot, also the mean cosine similairity of each of these words is shown.

    Parameters
    ----------
    vh : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the movies. It has dimensions (d,n_movies), where d=min{n_words,n_movies}. So, the
        columns correspond to the movies.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    df : pd.DataFrame
        Input dataframe
    genres : List[str]
        List of the genres of interest
    dimensions : Tuple[int], optional
        Two LSA dimensions onto which plot the words vectors, by default (0,1), by default (0,1)
    words : List[str], optional
        List of words to plot alongside the genres movies, by default None
    plot_most_relevant_words : bool, optional
        Whether to plot the most relevant words for each genre, by default False
    n : int, optional
        Number of most relevant words to plot for each genre, by default 1.
        This is used only if `plot_most_relevant_words` is True
    u : np.ndarray
        Bidimensional array, obtained from the SVD, by default None.
        It's the SVD matrix related to the words. It has dimensions (n_words,d), where d=min{n_words,n_movies}. So, the rows
        correspond to the words.
        It must be specified only if either `word` or `plot_most_relevant_words` are specified.
    voc : np.ndarray, optional
        Vocabulary, namely mapping from integer ids into words, by default None.
        It must be specified only if either `word` or `plot_most_relevant_words` are specified.
    normalize : bool, optional
        Whether to normalize or not the movies vectors and the words vectors, by default False
    k : int, optional
        Level of approximation for the LSA: k-rank approximation, by default 100. Basically, new number of dimensions.
        This is used only if either `plot_most_relevant_words` or `normalize` are True.
    delete_intersection : bool, optional
        Whether to delete or not the movies which belong to more than one of the specified genres, by default True
    """
    colors_indices = list(mcolors.TABLEAU_COLORS)

    if words is not None and (voc is None or u is None):
        raise ValueError('`words` is not None but either `u` or `voc` or both of them are None')
    if plot_most_relevant_words and (voc is None or u is None):
        raise ValueError('`plot_most_relevant_words` is True but either `u` or `voc` or both of them are None')
    if words is not None or plot_most_relevant_words:
        u_s = u * np.reshape(s, newshape=(1,s.shape[0]))
        if normalize:
            u_sk = u_s[:,:k]
            u_sk_normalized = u_sk/np.reshape(np.sqrt(np.sum(np.square(u_sk), axis=1)),newshape=(u_sk.shape[0],1))
            u_s = u_sk_normalized

    vh_s = np.reshape(s, newshape=(s.shape[0],1)) * vh
    if normalize:
        vh_sk = vh_s[:k,:]
        vh_sk_normalized = vh_sk/np.sqrt(np.sum(np.square(vh_sk), axis=0))
        vh_s = vh_sk_normalized

    dim_x = dimensions[0]
    dim_y = dimensions[1]

    if ax is None:
        w, h, dpi = 640, 640, 50
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)

    intersection_mask = df['genres'].map(lambda s: len(set(genres).intersection(s))>=2, na_action='ignore')
    intersection_mask = intersection_mask.map(lambda s: False if pd.isna(s) else s).to_numpy()
    for i, genre in enumerate(genres):
        color = mcolors.TABLEAU_COLORS[colors_indices[i]]
        genre_mask = df['genres'].map(lambda s: genre in s, na_action='ignore')
        genre_mask = genre_mask.map(lambda s: False if pd.isna(s) else s).to_numpy()

        if delete_intersection:
            genre_mask = np.logical_and(genre_mask, np.logical_not(intersection_mask))
        ax.scatter(vh_s[dim_x,genre_mask], vh_s[dim_y,genre_mask], label=f'{genre} movies', c=color)
        if plot_most_relevant_words:
            subset = np.arange(vh.shape[1])[genre_mask]
            selected_words_ids, mean_cos_similarities = find_most_interesting_words(vh, s, u, subset=subset, n=n, k=k,
                                                                                    normalize=normalize)
            ax.scatter(u_s[selected_words_ids,dim_x], u_s[selected_words_ids,dim_y], c=color, marker='*', #edgecolors='black',
                       s=100, label=f'{genre} words')
            for i, word_id in enumerate(selected_words_ids):
                txt = voc[word_id]
                mean_cos_similarity = mean_cos_similarities[i]
                ax.annotate(txt + f' {mean_cos_similarity:.2f}', (u_s[word_id,dim_x], u_s[word_id,dim_y])) 

    if words is not None:
        word2id = {word:id for id,word in enumerate(voc)}
        words_array = np.zeros(shape=(len(words),u.shape[1] if not normalize else k))
        for i, word in enumerate(words):  
            word_id = word2id[word]           
            words_array[i,:] = u_s[word_id,:]
        
        plt.scatter(words_array[:,dim_x], words_array[:,dim_y], c='red', marker='*', s=100, label='Specified words')
        for i in range(len(words)):
            txt = words[i]
            ax.annotate(txt, (words_array[i,dim_x], words_array[i,dim_y]))

    ax.scatter(0, 0, c='black', marker='s', label='origin')
    ax.set_xlabel(f'LSA dimension {dim_x}')
    ax.set_ylabel(f'LSA dimension {dim_y}')
    ax.set_title(f'Genre analysis in the LSA space along dimensions {dim_x} and {dim_y}')
    ax.grid()
    ax.legend()

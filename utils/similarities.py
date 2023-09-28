import numpy as np
import pandas as pd
from colorama import Fore, Style


def compute_cos_similarities(vector : np.ndarray, vectors : np.ndarray, vector_norm : float = None, 
                             vectors_norms : np.ndarray = None):
    """Compute the cosine similarities between the given vectors and each one of the column vectors in the given matrix.

    The specified `vector` is a row vector (d,), while the specified matrix `vectors` is (d,n): we want to compute the cosine
    similarity between the row vector and each one of the column vectors of the matrix.

    Parameters
    ----------
    vector : np.ndarray
        Input row vector
    vectors : np.ndarray
        Input matrix
    vector_norm : np.ndarray, optional
        Norm of the row vector, by default None
    vectors_norms : np.ndarray
        Norms of the column vectors of the matrix, by default None

    Returns
    -------
    np.ndarray
        Flat array containing the cosine similarities
    """
    if vector_norm is None:
        vector_norm = np.sqrt(np.sum(vector*vector))
    if vectors_norms is None:
        vectors_norms = np.sqrt(np.sum(vectors*vectors, axis=0))

    denominator = vector_norm*vectors_norms
    numerator = np.sum(vector[:,np.newaxis]*vectors, axis=0)
    return numerator/np.where(denominator!=0, denominator, 1)


def find_most_interesting_words(vt : np.ndarray, s : np.ndarray, u : np.ndarray, k : int = 100, subset : np.ndarray = None,
                                n : int = 3, normalize : bool = False):
    """Find the `n` most interesting words with respect to the given movies, according to the latent semantic analysis (i.e.
    LSA).

    The given movies are encoded as column vectors in the matrix `vh`; the words are encoded as row vectors in the matrix `u`.
    Optionally, we can consider only some of the movies, and not all of them (we can consider only a subset of the columns
    in `vh`). This can be useful, for example for focusing only to the movies with a certain genre.

    We want to find the `n` words which are the most related to the given movies. Namely, the words which are the most similar
    to the given movies.
    To be more specific, for each word, its average cosine similarity computed w.r.t. all the specified movies is calculated.

    Parameters
    ----------
    vt : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the movies. It has dimensions (d,n_movies), where d=min{n_words,n_movies}. So, the
        columns correspond to the movies.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    u : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the words. It has dimensions (n_words,d), where d=min{n_words,n_movies}. So, the rows
        correspond to the words.
    k : int, optional
        Level of approximation for the LSA: k-rank approximation, by default 100. Basically, new number of dimensions.
    subset : np.ndarray, optional
        Array of integers, containing the indicies of the movies in which we want to focus on, by default None
    n : int, optional
        Number of words to retrieve, by default 3
    normalize : bool, optional
        Whether to normalize or not the movies vectors and the words vectors, by default False

    Returns
    -------
    selected_words_ids : np.ndarray
        Array containing the ids of the selected words
    mean_cos_similarities : np.ndarray
        Array containing the mean cosine similarities of the selected words w.r.t all the specified movies
    """
    sk = s[:k]

    if subset is None:
        subset = range(vt.shape[1])
    vt_k = vt[:k,subset]
    vt_ks = np.reshape(np.sqrt(sk), newshape=(sk.shape[0],1)) * vt_k
    if normalize:
        vt_ks_normalized = vt_ks/np.sqrt(np.sum(np.square(vt_ks), axis=0))
        vt_ks = vt_ks_normalized

    u_k = u[:,:k]
    u_ks = u_k * np.reshape(np.sqrt(sk), newshape=(1,sk.shape[0]))
    if normalize:
        u_ks_normalized = u_ks/np.reshape(np.sqrt(np.sum(np.square(u_ks), axis=1)),newshape=(u_ks.shape[0],1))
        u_ks = u_ks_normalized

    vh_ks_norms = np.sqrt(np.sum(np.square(vt_ks),axis=0))
    u_ks_norms = np.sqrt(np.sum(np.square(u_ks),axis=1))

    mean_cos_similarities = np.zeros(shape=(u.shape[0],))
    for word_id in range(u_ks.shape[0]):
        cos_similarities = compute_cos_similarities(vector=u_ks[word_id,:], vectors=vt_ks, vector_norm=u_ks_norms[word_id],
                                                    vectors_norms=vh_ks_norms)
        cos_similarities = cos_similarities[~np.isnan(cos_similarities)]
        mean_cos_similarities[word_id] = np.mean(cos_similarities)

    selected_words_ids = np.argsort(mean_cos_similarities)[::-1]
    selected_words_ids = selected_words_ids[~np.isnan(mean_cos_similarities[selected_words_ids])]
    selected_words_ids = selected_words_ids[:n]

    return selected_words_ids, mean_cos_similarities[selected_words_ids]

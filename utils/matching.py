import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import jellyfish
from numba import jit
from collections import defaultdict


def damerau_levenshtein_distance(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    infinite = len1 + len2

    # character array
    da = defaultdict(int)

    score = []
    # distance matrix
    score = np.zeros((len1 + 2, len2 + 2))
    #score = [[0] * (len2 + 2) for x in range(len1 + 2)]

    score[0][0] = infinite
    for i in range(0, len1 + 1):
        score[i + 1][0] = infinite
        score[i + 1][1] = i
    for i in range(0, len2 + 1):
        score[0][i + 1] = infinite
        score[1][i + 1] = i

    for i in range(1, len1 + 1):
        db = 0
        for j in range(1, len2 + 1):
            i1 = da[s2[j - 1]]
            j1 = db
            cost = 1
            if s1[i - 1] == s2[j - 1]:
                cost = 0
                db = j

            score[i + 1][j + 1] = min(
                score[i][j] + cost,
                score[i + 1][j] + 1,
                score[i][j + 1] + 1,
                score[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
            )
        da[s1[i - 1]] = i

    return score[len1 + 1][len2 + 1]


#@jit
def compute_entry_distance(carte_entry, db_entry, maximize=False):
    # TODO: pass distance metric from outside

    a = np.zeros((len(carte_entry), len(db_entry)))
    for i, s1 in enumerate(carte_entry):
        for j, s2 in enumerate(db_entry):
            # a[i,j] = damerau_levenshtein_distance(s1, s2)
            a[i,j] = jellyfish.damerau_levenshtein_distance(s1, s2)

    if maximize:
        result = a.max(axis=np.argmax(a.shape)).mean()
    else:
        result = a.min(axis=np.argmax(a.shape)).mean()
    return result


def compute_entry_distance_weighted(carte_entry, db_entry, maximize=False):
    d = 0
    for key in carte_entry.keys():
        entry_1 = carte_entry[key]['tokens']

        key = key if key != 'territory' else 'region'
        entry_2 = db_entry[key]['tokens']

        d += compute_entry_distance(entry_1, entry_2, maximize) * carte_entry[key]['weight']
    return d


def match_data_edit_distance(df_carte: pd.DataFrame, df_database: pd.DataFrame, metric=None, maximize=False, weighted=True):
    """Return the matchig indices. For now of the same country.
    This is a dumb implementation, which just uses exaxt string matching.

    Parameters
    ----------
    df_carte : pd.DataFrame
        dataframe of the carte
    df_database : pd.DataFrame
        dataframe of the wine database
    metric : Callable[str, str] -> float, optional
        function that computes the distace between 2 strings.
        By default `jellyfish.damerau_levenshtein_distance`
    maximize : bool, optional
        if True, the metric needs to be maxmimized, otherwise minimized. By default False

    Returns
    -------
    dict
        keys are:
         - `'carte_idx'`: int, index of the wine in the carte
         - `'db_idx'`: list of dict, ranked matches for each wine in carte_idx
            - `'idx'`: int, index of the match in the database
            - `'score'`: float, matching score
    """

    preprocess_func = preprocess_data_weighted if weighted else preprocess_data
    compute_distance_func = compute_entry_distance_weighted if weighted else compute_entry_distance

    carte_preprocessed = preprocess_func(df_carte, is_carte=True)
    database_preprocessed = preprocess_func(df_database)

    doc_similarities = np.zeros((len(df_carte), len(df_database)))
    for i in tqdm(range(len(carte_preprocessed))):
        carte_entry = carte_preprocessed[i]
        for j, db_entry in enumerate(database_preprocessed):
            doc_similarities[i,j] = compute_distance_func(carte_entry, db_entry, maximize=maximize)
            #doc_similarities[i,j] = compute_entry_distance(carte_entry, db_entry, metric=metric, maximize=maximize)

    retrieved_indices_sorted = np.argsort(doc_similarities, axis=1)
    if maximize:
        retrieved_indices_sorted = retrieved_indices_sorted[:,::-1]

    matches = [{'carte_idx': idx} for idx in df_carte.index]
    for i, match in enumerate(matches):
        match['db_idx'] = []
        for j in range(5):
            idx = retrieved_indices_sorted[i][j]
            match['db_idx'].append({'idx': idx, 'score': doc_similarities[i][idx]})

    return matches


def low_rank_approximation(X, rank):
    U, sigma, VT = np.linalg.svd(X, full_matrices=False)

    U_k = U[:, :rank]
    sigma_k = sigma[:rank]
    VT_k = VT[:rank]

    return U_k @ np.diag(sigma_k) @ VT_k


def plot_matrices(doc_term_db, X_k):
    w, h, dpi = 800, 800, 100
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(w/dpi, h/dpi), dpi=dpi)#, gridspec_kw={'height_ratios': (2,2,1)})

    axs = axs.ravel()

    axs[0].imshow(doc_term_db, cmap='gray')
    axs[0].set_title("Doc term matrix")
    axs[0].set_ylabel('Doc')
    axs[0].set_xlabel('Term')

    axs[1].imshow(X_k, cmap='gray')
    axs[1].set_title("Low rank approximation")
    axs[1].set_xlabel('Term')

    axs[2].imshow(X_k@X_k.T, cmap='gray')
    axs[2].set_title("Doc similarity (database)")

    axs[3].imshow(X_k.T@X_k, cmap='gray')
    axs[3].set_title("Term similarity")

    fig.tight_layout()

    plt.show()


def match_data_lsa(df_carte: pd.DataFrame, df_database: pd.DataFrame, rank=10, plot=False):
    """Return the matchig indices using latent semantic analysis

    Parameters
    ----------
    df_carte : pd.DataFrame
        dataframe of the carte
    df_database : pd.DataFrame
        dataframe of the wine database

    Returns
    -------
    list of dict
        keys are:
         - `'carte_idx'`: int, index of the wine in the carte
         - `'db_idx'`: list of dict, ranked matches for each wine in carte_idx
            - `'idx'`: int, index of the match in the database
            - `'score'`: float, matching score
    """

    doc_term_db, dictionary = compute_doc_term_matrix(df_database)
    doc_term_carte = compute_doc_term_matrix(df_carte, dictionary)

    # low rank approximation of doc_term_db matrix
    X_k = low_rank_approximation(doc_term_db, rank)

    # compute doc similarities and normalize along the second axis
    doc_similarities = doc_term_carte @ X_k.T
    norms = np.linalg.norm(doc_similarities, axis=1)
    doc_similarities /= np.where(norms==0, 1, norms)[:, None]

    if plot:
        plot_matrices(doc_term_db, X_k, doc_similarities)

    retrieved_indices_sorted = np.argsort(doc_similarities, axis=1)[:,::-1]

    matches = [{'carte_idx': idx} for idx in df_carte.index]
    for i, match in enumerate(matches):
        match['db_idx'] = []
        for j in range(5):
            idx = retrieved_indices_sorted[i][j]
            match['db_idx'].append({'idx': idx, 'score': doc_similarities[i][idx]})

    return matches

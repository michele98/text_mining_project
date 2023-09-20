import os
import time
import pickle
import hashlib

from typing import Callable


def get_hash(f, *args, **kwargs):
    """Compute hash of function and its arguments using sha1."""

    # add positional args to keyword arguments
    arg_names = f.__code__.co_varnames[:f.__code__.co_argcount]
    for i in range(min(len(args), len(arg_names))):
        kwargs[arg_names[i]] = args[i]

    # write the name of the function and the name and
    content_string = f.__name__
    for arg_name in sorted(kwargs.keys()):
        arg = kwargs[arg_name]
        content_string += arg_name
        content_string += arg.__name__ if callable(arg) else str(arg)

    # handle additional *args that f might have
    for i in range(len(args)):
        if i < len(arg_names):
            continue
        arg = args[i]
        content_string += arg.__name__ if callable(arg) else str(arg)

    return hashlib.sha1(content_string.encode()).hexdigest()


def get_stored_hash(filename):
    with open(filename, 'rb') as handle:
        loaded_hash, result = pickle.load(handle)
    return loaded_hash


def cache(filename: str, f: Callable, *args, **kwargs):
    """Results are saved in .my_cache/<filename>.
    If the inputs of the results to cache change,
    the cached file is overwritten after the computation is done."""
    hashcode = get_hash(f, *args, **kwargs)

    if filename is None:
        filename = hashcode

    if not os.path.exists('.my_cache'):
        os.makedirs('.my_cache')
    filename = os.path.join('.my_cache', filename)

    if os.path.exists(filename):
        print(f'Loading cached result from: {filename}')
        with open(filename, 'rb') as handle:
            loaded_hash, result = pickle.load(handle)
        if loaded_hash == hashcode:
            print('Done')
            return result
        print('Hash changed, computing again\n')

    print('Computing')
    t0 = time.time()
    result = f(*args, **kwargs)
    t1 = time.time()
    print(f'Computaton done. Execution time: {t1-t0:.2f}s\n')

    print(f'Caching result in: {filename}')
    with open(filename, 'wb') as handle:
        pickle.dump((hashcode, result), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done')

    return result


def ucache(f: Callable, *args, **kwargs):
    """Unnamed cahce. Results are saved in .my_cache/<filename>
    where <filename> is the hash of the function and its arguments,
    as computed by `compute_hash`.

    Of course, compared to the normal `cache` function, previously cached
    results are never overwritten, because if the input changes, its hash
    does as well.
    """
    return cache(None, f, *args, **kwargs)

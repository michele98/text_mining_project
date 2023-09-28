import os
import time
import pickle
import hashlib

import ast
import inspect
import textwrap

from typing import Callable

CACHE_DIR = '.my_cache'

def remove_comments_and_docstrings(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Remove docstrings
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                node.body.pop(0)
            # Remove comments
            node.body = [n for n in node.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Str)]
    return ast.unparse(tree).strip()


def get_function_code_as_string(func):
    """Get the function code without trailing whitespaces, docstrings and comments"""
    source_code = inspect.getsource(func)
    return remove_comments_and_docstrings(textwrap.dedent(source_code))


def get_hash_function(f, *args, **kwargs):
    """Compute hash of function and its arguments using sha1."""

    # add positional args to keyword arguments
    arg_names = f.__code__.co_varnames[:f.__code__.co_argcount]
    for i in range(min(len(args), len(arg_names))):
        kwargs[arg_names[i]] = args[i]

    # write the name of the function in the content to hash
    content_string = get_function_code_as_string(f)
    for arg_name in sorted(kwargs.keys()):
        arg = kwargs[arg_name]
        content_string += arg_name
        content_string += get_function_code_as_string(arg) if callable(arg) else str(arg)

    # handle additional *args that f might have
    for i in range(len(args)):
        if i < len(arg_names):
            continue
        arg = args[i]
        content_string += get_function_code_as_string(arg) if callable(arg) else str(arg)

    return hashlib.sha1(content_string.encode()).hexdigest()


def get_hash(*args, **kwargs):
    """Compute hash of arguments using sha1.
    If the first argument is a function, the result of get_hash_function is returned."""
    if callable(args[0]):
        return get_hash_function(args[0], *args[1:], **kwargs)

    if len(kwargs)>0:
        raise ValueError('Keyword arguments require the first element to be a function')
    content_string = ''
    for arg in args:
        content_string += get_function_code_as_string(arg) if callable(arg) else str(arg)

    return hashlib.sha1(content_string.encode()).hexdigest()


def get_stored_hash(filename):
    with open(filename, 'rb') as handle:
        loaded_hash, result = pickle.load(handle)
    return loaded_hash


def cache(filename: str, f: Callable, *args, **kwargs):
    """Results are saved in cache_dir/<filename>.
    If the input values change, the cached file is overwritten
    after the computation is done."""
    hashcode = get_hash(f, *args, **kwargs)

    if filename is None:
        filename = hashcode

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    filename = os.path.join(CACHE_DIR, filename)

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
    results are never overwritten, because if the inputs change, their hash
    does as well.
    """
    return cache(None, f, *args, **kwargs)


def ext_cache(*args, **kwargs):
    """Create an empty file with the filename equal to the hash of the provided arguments.
    If it does not exist, an empty file with the name of the hash is created.
    ATTENTION! This function has side effects, so be careful when calling it."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    filename = os.path.join(CACHE_DIR, get_hash(*args, **kwargs))
    print(f'Check file: {filename}', end=' ')

    if os.path.exists(filename):
        print(f'already exists')
        return True
    with open(filename, 'w') as f:
        pass

    print(f'created')
    return False

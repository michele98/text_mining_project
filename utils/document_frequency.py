import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count


def split_set(input_set, n):
    # Calculate the base size of each subset
    input_list = list(input_set)
    base_size = len(input_list) // n

    # Calculate the number of sublists that will have one extra element
    num_larger_subsets = len(input_list) % n

    subsets = []
    start = 0

    for i in range(n):
        sublist_size = base_size + (1 if i < num_larger_subsets else 0)
        sublist = input_list[start:start+sublist_size]
        subsets.append(set(sublist))
        start += sublist_size

    return subsets


def _compute_document_frequency(df, vocabulary):
    document_frequency = {}

    for word in tqdm(vocabulary):
        document_frequency[word] = 0
        for bow in df['summary_bow']:
            if word in bow.keys():
                document_frequency[word]+=1
    return document_frequency


def _compute_document_frequency_queue(df, vocabulary, queue):
    queue.put(_compute_document_frequency(df, vocabulary))


def compute_document_frequency(df: pd.DataFrame, vocabulary, num_jobs=None): # spawns child processes
    """Compute the document frequency of each word in the vocabulary.
    To improve performance, this function is parallelized.
    The document frequency dictionary is returned in descending order."""

    if type(vocabulary) is not set:
        vocabulary = set(vocabulary)
    if num_jobs is None:
        num_jobs = cpu_count()

    print(f"Computing document frequency, num jobs: {num_jobs}")
    if num_jobs <= 1:
        return _compute_document_frequency(df, vocabulary)

    q = Queue()
    processes = []
    document_frequency = []
    for voc in split_set(vocabulary, num_jobs):
        p = Process(target=_compute_document_frequency_queue, args=(df, voc, q))
        processes.append(p)
        p.start()
    for p in processes:
        ret = q.get() # will block
        document_frequency.append(ret)
    for p in processes:
        p.join()
    document_frequency = {key: value for d in document_frequency for key, value in d.items()}
    document_frequency = dict(sorted(document_frequency.items(), key=lambda item: item[1], reverse=True))
    return document_frequency

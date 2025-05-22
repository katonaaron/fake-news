import gc
import math
import os
import threading
import time

import tagme
import torch
from more_itertools import chunked


def clean_gpu(device):
    gc.collect()
    if device != 'cpu':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def do_in_chunks_dict(func, l, chunk_size):
    res = {}
    for chunk in chunked(l, chunk_size):
        res |= func(chunk)
    return res


def do_in_chunks_list(func, l, chunk_size, desc=None):
    res = []
    for chunk in chunked(l, chunk_size):  # tqdm(chunked(l, chunk_size), desc=desc, total=math.ceil(len(l)/chunk_size)):
        res += func(chunk)
    return res


def do_in_chunks_tensor(func, l: list, chunk_size, *args):
    return torch.cat([func(chunk, *args) for chunk in chunked(l, chunk_size)])


def setup_env_var(var_name, value, description):
    os.environ[var_name] = value
    print("{}: {}".format(description, value))


def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


tagme_sem = threading.Semaphore(value=4)


# @sleep_and_retry
# @limits(calls=10, period=30) #s
def tagme_annotate(text, long_text=3):
    tagme.GCUBE_TOKEN = "f228d32f-ce3a-4c61-b8ce-da0a3f2e154d-843339462"
    # with tagme_sem:
    while True:
        response = tagme.annotate(text, long_text=long_text)
        if response:
            return response
        print("tagme returned no response")
        time.sleep(30)


def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

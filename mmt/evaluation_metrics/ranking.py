from __future__ import absolute_import
from collections import defaultdict
import threading
import multiprocessing

import numpy as np
from sklearn.metrics import average_precision_score

from ..utils import to_numpy


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False,
        n_threads=1):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = [0]

    def cmc_thread(start_index, stop_index):
        for i in range(start_index, stop_index):
            # Filter out the same id and same camera
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                     (gallery_cams[indices[i]] != query_cams[i]))
            if separate_camera_set:
                # Filter out samples from same camera
                valid &= (gallery_cams[indices[i]] != query_cams[i])
            if not np.any(matches[i, valid]): continue
            if single_gallery_shot:
                repeat = 10
                gids = gallery_ids[indices[i][valid]]
                inds = np.where(valid)[0]
                ids_dict = defaultdict(list)
                for j, x in zip(inds, gids):
                    ids_dict[x].append(j)
            else:
                repeat = 1
            for _ in range(repeat):
                if single_gallery_shot:
                    # Randomly choose one instance for each id
                    sampled = (valid & _unique_sample(ids_dict, len(valid)))
                    index = np.nonzero(matches[i, sampled])[0]
                else:
                    index = np.nonzero(matches[i, valid])[0]
                delta = 1. / (len(index) * repeat)
                for j, k in enumerate(index):
                    if k - j >= topk: break
                    if first_match_break:
                        ret[k - j] += 1
                        break
                    ret[k - j] += delta
            num_valid_queries[0] += 1
    if n_threads > 1:
        n_range = np.linspace(0, m, n_threads + 1).astype(int)
        threads = [threading.Thread(target=cmc_thread, args=(n_range[i], n_range[i+1],)) for i in range(n_threads)]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
    else:
        cmc_thread(0, m)

    if num_valid_queries[0] == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries[0]


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None, n_threads=1):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    aps = []
    # Compute AP for each query using multi threads
    if n_threads > 1:
        def run_thread(start_index, stop_index):
            for i in range(start_index, stop_index):
                # Filter out the same id and same camera
                valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                         (gallery_cams[indices[i]] != query_cams[i]))
                y_true = matches[i, valid]
                y_score = -distmat[i][indices[i]][valid]
                if not np.any(y_true): continue
                aps.append(average_precision_score(y_true, y_score))

        n_range = np.linspace(0, m, n_threads+1).astype(int)
        threads = [threading.Thread(target=run_thread, args=(n_range[i], n_range[i+1])) for i in range(n_threads)]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
    else: #single thread
        for i in range(m):
            # Filter out the same id and same camera
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                     (gallery_cams[indices[i]] != query_cams[i]))
            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]
            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))

    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)




# from __future__ import absolute_import
# from collections import defaultdict

# import numpy as np
# import multiprocessing as mp
# from sklearn.metrics import average_precision_score

# from ..utils import to_numpy


# def _unique_sample(ids_dict, num):
#     mask = np.zeros(num, dtype=np.bool)
#     for _, indices in ids_dict.items():
#         i = np.random.choice(indices)
#         mask[i] = True
#     return mask

# ret = None
# num_valid_queries = None
# aps = None

# gallery_ids2 = None
# query_ids2 = None
# gallery_cams2 = None
# query_cams2 = None
# indices2 = None
# matches2 = None
# distmat2 = None


# def cmc_func(argument): # the results is not right
#     start_index, stop_index, separate_camera_set, single_gallery_shot, first_match_break, topk = argument
#     global num_valid_queries
#     global ret
#     for i in range(start_index, stop_index):
#         # Filter out the same id and same camera
#         valid = ((gallery_ids2[indices2[i]] != query_ids2[i]) |
#                     (gallery_cams2[indices2[i]] != query_cams2[i]))
#         if separate_camera_set:
#             # Filter out samples from same camera
#             valid &= (gallery_cams2[indices2[i]] != query_cams2[i])
#         if not np.any(matches2[i, valid]): continue
#         if single_gallery_shot:
#             repeat = 10
#             gids = gallery_ids2[indices2[i][valid]]
#             inds = np.where(valid)[0]
#             ids_dict = defaultdict(list)
#             for j, x in zip(inds, gids):
#                 ids_dict[x].append(j)
#         else:
#             repeat = 1
#         for _ in range(repeat):
#             if single_gallery_shot:
#                 # Randomly choose one instance for each id
#                 sampled = (valid & _unique_sample(ids_dict, len(valid)))
#                 index = np.nonzero(matches2[i, sampled])[0]
#             else:
#                 index = np.nonzero(matches2[i, valid])[0]
#             delta = 1. / (len(index) * repeat)
#             for j, k in enumerate(index):
#                 if k - j >= topk: break
#                 if first_match_break:
#                     ret[k - j] += 1
#                     break
#                 ret[k - j] += delta
#         num_valid_queries[0] += 1

# def cmc(distmat, query_ids=None, gallery_ids=None,
#         query_cams=None, gallery_cams=None, workers=0, topk=100,
#         separate_camera_set=False,
#         single_gallery_shot=False,
#         first_match_break=False):
#     distmat = to_numpy(distmat)
#     m, n = distmat.shape

#     # Fill up default values
#     if query_ids is None:
#         query_ids = np.arange(m)
#     if gallery_ids is None:
#         gallery_ids = np.arange(n)
#     if query_cams is None:
#         query_cams = np.zeros(m).astype(np.int32)
#     if gallery_cams is None:
#         gallery_cams = np.ones(n).astype(np.int32)
#     # Ensure numpy array
#     query_ids = np.asarray(query_ids)
#     gallery_ids = np.asarray(gallery_ids)
#     query_cams = np.asarray(query_cams)
#     gallery_cams = np.asarray(gallery_cams)
#     # Sort and find correct matches
#     indices = np.argsort(distmat, axis=1)
#     matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

#     global query_ids2
#     global gallery_ids2
#     global query_cams2
#     global gallery_cams2
#     global indices2
#     global matches2
#     global distmat2
#     global ret
#     global num_valid_queries

#     # Compute CMC for each query
#     ret = np.zeros(topk)
#     num_valid_queries = [0]


#     query_ids2 = query_ids
#     gallery_ids2 = gallery_ids
#     query_cams2 = query_cams
#     gallery_cams2 = gallery_cams
#     indices2 = indices
#     matches2 = matches
#     distmat2 = distmat


#     if workers > 0:
#         manager = mp.Manager()
#         ret = manager.list(ret) # items get write 'main location'
#         num_valid_queries = manager.list([0]) # items obey date rules
#         n_range = np.linspace(0, m, workers + 1).astype(int)
#         arguments = [(n_range[i], n_range[i+1], separate_camera_set, single_gallery_shot, first_match_break, topk) for i in range(workers)]
#         p = mp.Pool(workers)
#         p.map(cmc_func, arguments)
#         p.close()
#         p.join()
#     else:
#         cmc_func([0, m, separate_camera_set, single_gallery_shot, first_match_break, topk])

#     if num_valid_queries[0] == 0:
#         raise RuntimeError("No valid query")
#     ret = np.array(list(ret))
#     return ret.cumsum() / num_valid_queries[0]


# def run_func(argument):
#     global aps
#     start_index, stop_index = argument
#     for i in range(start_index, stop_index):
#         # Filter out the same id and same camera
#         valid = ((gallery_ids2[indices2[i]] != query_ids2[i]) |
#                     (gallery_cams2[indices2[i]] != query_cams2[i]))
#         y_true = matches2[i, valid]
#         y_score = -distmat2[i][indices2[i]][valid]
#         if not np.any(y_true): continue
#         aps.append(average_precision_score(y_true, y_score))

# def mean_ap(distmat, query_ids=None, gallery_ids=None,
#             query_cams=None, gallery_cams=None, workers=0):
#     distmat = to_numpy(distmat)
#     m, n = distmat.shape
#     # Fill up default values
#     if query_ids is None:
#         query_ids = np.arange(m)
#     if gallery_ids is None:
#         gallery_ids = np.arange(n)
#     if query_cams is None:
#         query_cams = np.zeros(m).astype(np.int32)
#     if gallery_cams is None:
#         gallery_cams = np.ones(n).astype(np.int32)
#     # Ensure numpy array
#     query_ids = np.asarray(query_ids)
#     gallery_ids = np.asarray(gallery_ids)
#     query_cams = np.asarray(query_cams)
#     gallery_cams = np.asarray(gallery_cams)

#     global query_ids2
#     global gallery_ids2
#     global query_cams2
#     global gallery_cams2
#     global indices2
#     global matches2
#     global distmat2

#     # Sort and find correct matches
#     indices = np.argsort(distmat, axis=1)
#     matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

#     query_ids2 = query_ids
#     gallery_ids2 = gallery_ids
#     query_cams2 = query_cams
#     gallery_cams2 = gallery_cams
#     indices2 = indices
#     matches2 = matches
#     distmat2 = distmat

#     global aps
#     aps = []
#     if workers > 1:
#         manager = mp.Manager()
#         aps = manager.list()
#         n_range = np.linspace(0, m, workers+1).astype(int)
#         arguments = [(n_range[i], n_range[i+1]) for i in range(workers)]
#         p = mp.Pool(workers)
#         p.map(run_func, arguments)
#         p.close()
#         p.join()
#         aps = list(aps)
#     else:
#         for i in range(m):
#             # Filter out the same id and same camera
#             valid = ((gallery_ids[indices[i]] != query_ids[i]) |
#                      (gallery_cams[indices[i]] != query_cams[i]))
#             y_true = matches[i, valid]
#             y_score = -distmat[i][indices[i]][valid]
#             if not np.any(y_true): continue
#             aps.append(average_precision_score(y_true, y_score))

#     if len(aps) == 0:
#         raise RuntimeError("No valid query")
#     aps = np.array(list(aps))
#     return np.mean(aps)

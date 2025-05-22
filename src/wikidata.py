import threading
import time

from SPARQLWrapper import SPARQLWrapper, JSON, POST

import shelve

USER_AGENT = "FakeNewsDetectorBot/0.0 (https://isg.utcluj.ro/; katona.io.aron@student.utcluj.ro)"

label_shelve = shelve.open('labels.shelf')
label_shelve['Q0'] = '<interaction>'
label_shelve['P0'] = '<input>'
label_shelve_lock = threading.Lock()


def get_labels_dict_from_wikidata_ids(wikidata_ids):
    result = {}
    remaining = []
    with label_shelve_lock:
        for wd_id in wikidata_ids:
            if wd_id in label_shelve:
                result[wd_id] = label_shelve[wd_id]
            else:
                remaining.append(wd_id)

    if len(remaining) == 0:
        return result

    query = f"""
        SELECT ?item ?itemLabel
        WHERE
        {{
          VALUES ?item {{ {" ".join(['wd:' + wd_id for wd_id in remaining])} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """

    sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)

    while True:
        try:
            ret = sparql.queryAndConvert()

            retResults = {row['item']['value'].rpartition('/')[2]: row['itemLabel']['value'] for row in
                          ret["results"]["bindings"]}

            with label_shelve_lock:
                for key, value in retResults.items():
                    label_shelve[key] = value
            return result | retResults

        except Exception as e:
            print(e)
            time.sleep(30)


def get_property_labels_dict_from_wikidata_ids(wikidata_ids):
    result = {}
    remaining = []
    with label_shelve_lock:
        for wdt_id in wikidata_ids:
            if wdt_id in label_shelve:
                result[wdt_id] = label_shelve[wdt_id]
            else:
                remaining.append(wdt_id)

    if len(remaining) == 0:
        return result

    query = f"""
        SELECT ?wdt ?wdLabel 
        WHERE 
        {{
          VALUES ?wdt {{ {" ".join(['wdt:' + wdt_id for wdt_id in remaining])} }}
          ?wd wikibase:directClaim ?wdt .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}

        """

    sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)

    while True:
        try:
            ret = sparql.queryAndConvert()

            retResults = {row['wdt']['value'].rpartition('/')[2]: row['wdLabel']['value'] for row in
                          ret["results"]["bindings"]}

            with label_shelve_lock:
                for key, value in retResults.items():
                    label_shelve[key] = value
            return result | retResults

        except Exception as e:
            print(e)
            time.sleep(30)


def get_labels_from_wikidata_ids(wikidata_ids, chunk_size=None):
    labels_dict = get_labels_dict_from_wikidata_ids(wikidata_ids)
    return _map_list_values(labels_dict, wikidata_ids)


def _map_list_values(labels_dict, items):
    return list(map(labels_dict.get, items))


def _get_labels_from_wikidata_ids(wikidata_ids):
    query = f"""
        SELECT ?itemLabel
        WHERE
        {{
          VALUES ?item {{ {" ".join(['wd:' + wd_id for wd_id in wikidata_ids])} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """

    sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)

    while True:
        try:
            ret = sparql.queryAndConvert()

            return [row['itemLabel']['value'] for row in ret["results"]["bindings"]]
        except Exception as e:
            print(e)
            time.sleep(30)


def get_wikidata_ids_from_uris(uris):
    query = f"""
        SELECT ?item
        WHERE
        {{
          VALUES ?url {{ {" ".join(['<' + uri + '>' for uri in uris])} }}
          ?url schema:about ?item.
        }}
        """

    print(query)

    sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql", agent=USER_AGENT)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    sparql.setMethod(POST)

    while True:
        try:
            ret = sparql.queryAndConvert()

            return [row['item']['value'].rpartition('/')[2] for row in ret["results"]["bindings"]]
        except Exception as e:
            print(e)
            time.sleep(30)


def get_wikidata_id_map_from_uris(uris):
    query = f"""
        SELECT ?url ?item
        WHERE
        {{
          VALUES ?url {{ {" ".join(['<' + uri + '>' for uri in uris])} }}
          ?url schema:about ?item.
        }}
        """

    sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql", agent=USER_AGENT)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    sparql.setMethod(POST)

    while True:
        try:
            ret = sparql.queryAndConvert()

            return {row['url']['value']: row['item']['value'].rpartition('/')[2] for row in ret["results"]["bindings"]}
        except Exception as e:
            print(e)
            time.sleep(30)


def get_wikidata_ids_and_labels_from_uris(uris):
    query = f"""
        SELECT ?item ?itemLabel
        WHERE
        {{
          VALUES ?url {{ {" ".join(['<' + uri + '>' for uri in uris])} }}
          ?url schema:about ?item.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """

    sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    sparql.setMethod(POST)

    while True:
        try:
            ret = sparql.queryAndConvert()

            return [row['item']['value'].rpartition('/')[2] for row in ret["results"]["bindings"]], [
                row['itemLabel']['value'] for row in ret["results"]["bindings"]]
        except Exception as e:
            print(e)
            time.sleep(30)

# import threading
# import time
#
# import requests
# from SPARQLWrapper import SPARQLWrapper, JSON, POST
#
# import shelve
#
# from more_itertools.more import chunked
#
# from util import do_in_chunks_dict
#
# label_shelve_lock = threading.Lock()
# label_shelve = shelve.open('labels.shelf')
# label_shelve['Q0'] = '<interaction>'
#
# uri_shelve_lock = threading.Lock()
# uri_shelve = shelve.open('uris.shelf')
#
#
# def _get_labels_dict_from_wikidata_ids(wikidata_ids):
#     query = f"""
#         SELECT ?item ?itemLabel
#         WHERE
#         {{
#           VALUES ?item {{ {" ".join(['wd:' + wd_id for wd_id in wikidata_ids])} }}
#           SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
#         }}
#         """
#
#     sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")
#     sparql.setReturnFormat(JSON)
#     sparql.setQuery(query)
#
#     while True:
#         try:
#             ret = sparql.queryAndConvert()
#
#             return {row['item']['value'].rpartition('/')[2]: row['itemLabel']['value'] for row in
#                     ret["results"]["bindings"]}
#
#         except Exception as e:
#             print(e)
#             time.sleep(30)
#
#
# # def _return(cached_results, results, items):
# #     if isinstance(results, list):
# #         return _map_list_values(cached_results, items)
# #     else:
# #         return cached_results | results
# def _execute_cached(func, items, cache, cache_lock, chunk_size=None):
#     cached_results = {}
#     remaining = []
#     with cache_lock:
#         for item in items:
#             if item in cache:
#                 cached_results[item] = cache[item]
#             else:
#                 remaining.append(item)
#
#     if len(remaining) == 0:
#         return cached_results
#
#     if chunk_size is None:
#         results = func(remaining)
#     else:
#         results = do_in_chunks_dict(func, remaining, chunk_size=chunk_size)
#
#         results = {}
#         for chunk in chunked(remaining, chunk_size):
#             res = func(chunk)
#             with cache_lock:
#                 for key, value in res.items():
#                     cache[key] = value
#             results |= res
#         return results
#
#     # if isinstance(results, list):
#     #     for idx, value in enumerate(results):
#     #         cache[remaining[idx]] = value
#     #         cached_results[remaining[idx]] = value
#     #     return _map_list_values(cached_results, items)
#     # else:
#
#     return cached_results | results
#
#
# def get_labels_dict_from_wikidata_ids(wikidata_ids, chunk_size=None):
#     return _execute_cached(_get_labels_dict_from_wikidata_ids, wikidata_ids, label_shelve, label_shelve_lock, chunk_size)
#
#
# def get_labels_from_wikidata_ids(wikidata_ids, chunk_size=None):
#     labels_dict = get_labels_dict_from_wikidata_ids(wikidata_ids, chunk_size)
#     return _map_list_values(labels_dict, wikidata_ids)
#
#
# def _map_list_values(labels_dict, items):
#     return list(map(labels_dict.get, items))
#
#
# def _get_labels_from_wikidata_ids(wikidata_ids):
#     query = f"""
#         SELECT ?itemLabel
#         WHERE
#         {{
#           VALUES ?item {{ {" ".join(['wd:' + wd_id for wd_id in wikidata_ids])} }}
#           SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
#         }}
#         """
#
#     sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")
#     sparql.setReturnFormat(JSON)
#     sparql.setQuery(query)
#
#     while True:
#         try:
#             ret = sparql.queryAndConvert()
#
#             return [row['itemLabel']['value'] for row in ret["results"]["bindings"]]
#         except Exception as e:
#             print(e)
#             time.sleep(30)
#
#
# def _get_wikidata_ids_from_uris(uris):
#     query = f"""
#         SELECT ?item
#         WHERE
#         {{
#           VALUES ?url {{ {" ".join(['<' + uri + '>' for uri in uris])} }}
#           ?url schema:about ?item.
#         }}
#         """
#
#     sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")
#     sparql.setReturnFormat(JSON)
#     sparql.setQuery(query)
#     sparql.setMethod(POST)
#
#     while True:
#         try:
#             ret = sparql.queryAndConvert()
#
#             return [row['item']['value'].rpartition('/')[2] for row in ret["results"]["bindings"]]
#         except Exception as e:
#             print(e)
#             time.sleep(30)
#
#
# def _get_wikidata_ids_dict_from_uris(uris):
#     query = f"""
#         SELECT ?url ?item
#         WHERE
#         {{
#           VALUES ?url {{ {" ".join(['<' + uri + '>' for uri in uris])} }}
#           ?url schema:about ?item.
#         }}
#         """
#
#     sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")
#     sparql.setReturnFormat(JSON)
#     sparql.setQuery(query)
#     sparql.setMethod(POST)
#
#     while True:
#         try:
#             ret = sparql.queryAndConvert()
#
#             return {row['url']['value']: row['item']['value'].rpartition('/')[2] for row in
#                     ret["results"]["bindings"]}
#         except Exception as e:
#             print(e)
#             time.sleep(30)
#
#
# def get_wikidata_ids_from_uris(uris, chunk_size=None):
#     id_dict = _execute_cached(_get_wikidata_ids_dict_from_uris, uris, uri_shelve, uri_shelve_lock, chunk_size)
#     return _map_list_values(id_dict, uris)
#
#
# def get_wikidata_ids_and_labels_from_uris(uris):
#     query = f"""
#         SELECT ?item ?itemLabel
#         WHERE
#         {{
#           VALUES ?url {{ {" ".join(['<' + uri + '>' for uri in uris])} }}
#           ?url schema:about ?item.
#           SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
#         }}
#         """
#
#     sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")
#     sparql.setReturnFormat(JSON)
#     sparql.setQuery(query)
#     sparql.setMethod(POST)
#
#     while True:
#         try:
#             ret = sparql.queryAndConvert()
#
#             return [row['item']['value'].rpartition('/')[2] for row in ret["results"]["bindings"]], [
#                 row['itemLabel']['value'] for row in ret["results"]["bindings"]]
#         except Exception as e:
#             print(e)
#             time.sleep(30)
#
#
# def get_wikidata_id_from_wikipedia_title(title: str) -> str | None:
#     r = requests.get(
#         f'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&format=json&titles={title}')
#     try:
#         pages = r.json()['query']['pages']
#         page_id = next(iter(pages))
#         return pages[page_id]['pageprops']['wikibase_item']
#     except AttributeError:
#         return None

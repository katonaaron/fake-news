import copy
import json
import os.path as osp
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tagme
import torch
# from adapters import BertAdapterModel
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from src.util import do_in_chunks_list, swap_columns, tagme_annotate
from src.wikidata import get_labels_from_wikidata_ids, get_wikidata_id_map_from_uris

LOG_COLUMNS = ['id', 'title', 'label', 'skipped', 'has_content', 'count_title_annotations', 'count_content_annotations',
               'value_counts_before_trim', 'count_nodes', 'count_edges',  # 'title_annotations', 'content_annotations',
               'wikidata_ids']

RHO_TITLE = 0.08
RHO_TEXT = 0.1
TEXT_ANNOTATE_WINDOW = 1  # 3
# TEXT_MAX_ANNOTATIONS = 200
TAGME_MAX_TEXT_LENGTH = 700

MAX_NODES = 1000  # titles 200
MAX_NEIGHBOURS = 7  # 5

mutex = threading.Lock()
# gpu_mutex = threading.Lock()
gpu_lock = threading.Semaphore(value=2)


def subtract(df1, df2):
    return pd.concat([df1, df2, df2]).drop_duplicates(keep=False)


def id_to_filename(row_id):
    return f'data_{row_id}.pt'


class FNNWikidata5MDataset(Dataset):
    def __init__(self, root, dataset, device, news_content=True, skip_processing=False, transform=None,
                 pre_transform=None, pre_filter=None):
        self.all_triples = None
        self.dataset = dataset
        self.device = device
        self.data = None
        self._file_names = None
        self.tokenizer = None
        self.encoder = None
        self.skip_processing = skip_processing
        self.news_content = news_content
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['politifact_fake.csv', 'politifact_fake_news_content.csv', 'politifact_real.csv',
                'politifact_real_news_content.csv', 'wikidata5m_all_triplet.txt', 'wikidata5m_entity.txt',
                'wikidata5m_relation.txt']

    @property
    def processed_file_names(self):
        self._read_data()
        return self._file_names  # return os.listdir(self.processed_dir)

    @property
    def log_path(self):
        return osp.join(self.processed_dir, f"{self.dataset}_log.csv")

    @property
    def temp_dir(self):
        return osp.join(self.processed_dir, f"temp")

    @property
    def train_dir(self):
        return osp.join(self.processed_dir, 'train')

    @property
    def val_dir(self):
        return osp.join(self.processed_dir, 'val')

    @property
    def test_dir(self):
        return osp.join(self.processed_dir, 'test')

    def download(self):
        pass

    def len(self):
        # return self.data.shape[0]
        return len(self.processed_paths)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data

    def process(self):
        filename_all_triples = 'wikidata5m_all_triplet.txt'
        self.all_triples = pd.read_csv(osp.join(self.raw_dir, 'wikidata5m', filename_all_triples),
                                       names=['s', 'p', 'o'], delimiter='\t')

        data = self.data
        data_size = data.shape[0]

        if data_size == 0:
            return

        print(f"Processing {data_size} entries", file=sys.stderr)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.encoder.eval()
        # self.encoder = BertAdapterModel.from_pretrained('bert-base-uncased')
        #
        # # Load pre-trained task adapter from Adapter Hub
        # # This method call will also load a pre-trained classification head for the adapter task
        # adapter_name = self.encoder.load_adapter("sentiment/sst-2@ukp", config='pfeiffer')
        #
        # # Activate the adapter we just loaded, so that it is used in every forward pass
        # self.encoder.set_active_adapters(adapter_name)

        # num_threads = min(multiprocessing.cpu_count(), data_size)
        num_threads = 1

        chunks = np.array_split(data, num_threads)
        del data, self.data

        Path(osp.join(self.processed_dir, 'annotations')).mkdir(parents=True, exist_ok=True)

        threads = [threading.Thread(target=self._process_chunk, args=(chunks[i], i,)) for i in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self._set_file_names_from_log()
        self.data = None

    def _read_data(self):
        if self._file_names is not None:
            return

        if self.skip_processing:
            self._set_file_names_from_log()
            return

        log_path = self.log_path
        if osp.exists(log_path):
            log = pd.read_csv(log_path)
        else:
            log = pd.DataFrame([], columns=LOG_COLUMNS)
            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
            log.to_csv(log_path, mode='w', index=False, header=True)

        filename_real = self.dataset + '_real.csv'
        filename_fake = self.dataset + '_fake.csv'

        ds_real = pd.read_csv(osp.join(self.raw_dir, filename_real))
        ds_fake = pd.read_csv(osp.join(self.raw_dir, filename_fake))
        ds_real["label"] = 0
        ds_fake["label"] = 1

        self.data = pd.concat([ds_real, ds_fake]).reset_index(drop=True).drop_duplicates(subset='id', keep=False)

        self.data.drop(self.data[self.data.id.isin(set(log['id'].to_list()))].index, inplace=True)

        if self.data.shape[0] == 0:
            self.data = None
            self._set_file_names_from_log()
        else:
            self._file_names = self.data['id'].map(id_to_filename).to_list()

    def _set_file_names_from_log(self):
        log = pd.read_csv(self.log_path)
        log = log[~log['skipped']]

        if self.news_content:
            log = log[log['has_content']]

        if log.shape[0] == 0:
            self._file_names = []
        else:
            self._file_names = log['id'].map(id_to_filename).to_list()

    def _process_chunk(self, chunk, thread_id):
        tagme.GCUBE_TOKEN = "f228d32f-ce3a-4c61-b8ce-da0a3f2e154d-843339462"

        total_rows = chunk.shape[0]
        for index, row in tqdm(chunk.iterrows(), total=total_rows, desc="Processing chunk #{}".format(thread_id)):

            annotations_path = osp.join(self.processed_dir, 'annotations', row['id'] + '.csv')
            if osp.isfile(annotations_path):
                annotations = pd.read_csv(annotations_path)
                title_annotations = annotations[annotations['title']]
                content_annotations = annotations[annotations['content']]
                has_content = content_annotations.size > 0
            else:
                # log.write(f"[{thread_id}] Processing {index}: {row['title']}\n")
                response = tagme_annotate(row['title'])
                if response is None:
                    print("No tagme response", file=sys.stderr)
                    continue
                title_annotations = list(response.get_annotations(RHO_TITLE))

                content_annotations = []
                has_content = False
                if self.news_content:
                    if row['label'] == 0:
                        label = 'real'
                    else:
                        label = 'fake'

                    content_annotations = []
                    content_path = osp.join(self.raw_dir, self.dataset, label, row['id'], 'news content.json')
                    if osp.isfile(content_path):

                        with open(content_path) as f:
                            news_content = json.load(f)
                            if news_content['text']:
                                has_content = True
                                content_text = news_content['text']
                                try:
                                    # content_text = re.sub(r'(^\s+)|(\s+$)', '', content_text)
                                    # paragraphs = re.split(r'[\n\r]+', content_text)
                                    # print(
                                    #     f"[{thread_id}][{datetime.now().time()}] Nr paragraphs: {len(paragraphs)})\n")
                                    #
                                    # for paragraph in paragraphs:
                                    #     response = tagme_annotate(paragraph, long_text=TEXT_ANNOTATE_WINDOW)
                                    #     if response is None:
                                    #         print("No tagme response: " + row['id'], file=sys.stderr)
                                    #         continue
                                    #     paragraph_annotations = list(response.get_annotations(
                                    #         RHO_TEXT))
                                    #     if paragraph_annotations:
                                    #         content_annotations += paragraph_annotations

                                    print(f"[{thread_id}][{datetime.now().time()}] {row['id']} : {len(content_text)}\n")

                                    remaining_text = content_text
                                    while remaining_text:
                                        current_text = remaining_text
                                        if len(current_text) > TAGME_MAX_TEXT_LENGTH:
                                            index = current_text.rfind(' ', 0, TAGME_MAX_TEXT_LENGTH - 1)
                                            if index < 0:
                                                print("cannot split: " + row['id'], file=sys.stderr)
                                                continue
                                            else:
                                                remaining_text = current_text[index:]
                                                current_text = current_text[:index]
                                        else:
                                            remaining_text = None

                                        response = tagme_annotate(current_text, long_text=TEXT_ANNOTATE_WINDOW)
                                        if response is None:
                                            print("No tagme response: " + row['id'], file=sys.stderr)
                                            # continue
                                            exit(1)

                                        content_annotations += list(response.get_annotations(RHO_TEXT))

                                    # content_annotations = list(response.get_annotations(RHO_TEXT))  # if len(content_annotations) > 0:  #     content_annotations.sort(key=lambda ann: ann.score,  #                              reverse=True)  # if len(content_annotations) > TEXT_MAX_ANNOTATIONS:  #     content_annotations = content_annotations[:TEXT_MAX_ANNOTATIONS]
                                except Exception as e:
                                    print(row['id'], e, content_text)
                                    continue

                print(f"[{thread_id}][{datetime.now().time()}] {row['id']} annotated\n")

                title_annotations = pd.DataFrame([[ann.uri(), ann.score] for ann in title_annotations],
                                                 columns=['uri', 'score'])
                content_annotations = pd.DataFrame([[ann.uri(), ann.score] for ann in content_annotations],
                                                   columns=['uri', 'score'])

                if content_annotations.size == 0:
                    annotations = title_annotations
                elif title_annotations.size == 0:
                    annotations = content_annotations
                else:
                    annotations = pd.concat([title_annotations, content_annotations], ignore_index=True)

                annotations.sort_values(by=['score'], ascending=False, inplace=True)
                annotations.drop_duplicates(subset=['uri'], inplace=True, ignore_index=True)

                annotations['title'] = annotations['uri'].isin(title_annotations['uri'])
                annotations['content'] = annotations['uri'].isin(content_annotations['uri'])

                annotations.to_csv(annotations_path, mode='w', index=False)

            uris = set(annotations['uri'])
            # log.write(f"Uris: {uris}\n")

            if len(uris) == 0:
                # log.write(f"No annotations found\n")
                # log.write("\n")
                # log.flush()
                self._log_iteration(row['id'], row['title'], row['label'], skipped=True, has_content=has_content)
                continue

            wikidata_uri_to_id = get_wikidata_id_map_from_uris(uris)

            wikidata_ids = set(wikidata_uri_to_id.values())
            if None in wikidata_ids:
                wikidata_ids.remove(None)
            # log.write(f"Wikidata ids: {wikidata_ids}\n")

            if len(wikidata_ids) == 0:
                self._log_iteration(row['id'], row['title'], row['label'], skipped=True, has_content=has_content,
                                    title_annotations=title_annotations, content_annotations=content_annotations)
                continue

            # del annotations[~annotations.uri.isin(wikidata_uri_to_id.keys())]

            edges = self.all_triples[(self.all_triples.s.isin(wikidata_ids)) | (self.all_triples.o.isin(wikidata_ids))]

            # log.write(f"Edges: {edges.shape[0]}\n")
            # log.write("\n")
            # log.flush()

            # nr_edges = edges.shape[0]
            # if nr_edges == 0:
            #     self._log_iteration(row['id'], row['title'], row['label'], skipped=True,
            #                         title_annotations=title_annotations, content_annotations=content_annotations)
            #     continue
            #

            detected_entities = pd.Series(list(wikidata_ids))
            base_node_ids = pd.concat([pd.Series(['Q0']), detected_entities])

            print(
                f"[{thread_id}][{datetime.now().time()}] Start processing entities (base nodes: {base_node_ids.shape[0]})\n")

            # # limit the number of nodes
            # entity_aparitions = pd.concat([edges.s, edges.o]).value_counts()
            # value_counts_before_trim = entity_aparitions.value_counts().sort_values(ascending=False)
            #
            # nr_secondary_nodes = min(entity_aparitions.shape[0], MAX_NODES) - base_node_ids.shape[0]
            # if nr_secondary_nodes > 0:
            #     secondary_nodes = entity_aparitions[~entity_aparitions.index.isin(base_node_ids)].sort_values(
            #         ascending=False)[:nr_secondary_nodes].index.to_series()
            #     entities = pd.concat([base_node_ids, secondary_nodes]).reset_index(drop=True)
            #     del secondary_nodes
            # else:
            #     entities = base_node_ids
            #
            # del entity_aparitions
            #
            # edges = edges[edges.s.isin(entities) & edges.o.isin(entities)]

            # limit the number of nodes
            entity_scores = pd.Series(annotations['score'].values, index=annotations['uri'].map(wikidata_uri_to_id.get))
            if None in entity_scores.index:
                del entity_scores[None]
            entity_apparitions = pd.concat([edges.s, edges.o]).value_counts()
            value_counts_before_trim = entity_apparitions.value_counts().sort_values(ascending=False)

            # nr_secondary_nodes = min(entity_apparitions.size, MAX_NODES) - base_node_ids.size
            # if nr_secondary_nodes > 0:
            #     secondary_nodes = entity_apparitions[~entity_apparitions.index.isin(base_node_ids)]
            #     secondary_nodes = pd.DataFrame({'id':secondary_nodes.index, 'apparitions':secondary_nodes.values})
            #     secondary_nodes['score'] = secondary_nodes.id.map(e)
            #
            #     .sort_values(
            #         ascending=False)[:nr_secondary_nodes].index.to_series()
            #     entities = pd.concat([base_node_ids, secondary_nodes]).reset_index(drop=True)
            #     del secondary_nodes
            # else:
            #     entities = base_node_ids
            #
            # edges = edges[edges.s.isin(entities) & edges.o.isin(entities)]
            #     entities = pd.concat([base_node_ids, secondary_nodes]).reset_index(drop=True)

            # nr_secondary_nodes = min(entity_aparitions.shape[0], MAX_NODES) - base_node_ids.shape[0]
            # if nr_secondary_nodes > 0:
            #     secondary_nodes = entity_aparitions[~entity_aparitions.index.isin(base_node_ids)].sort_values(
            #         ascending=False)
            edges_base_to_secondary = pd.concat(
                [edges[edges.s.isin(base_node_ids) & (~edges.o.isin(base_node_ids))].assign(p_reverse='0'),
                 edges[edges.o.isin(base_node_ids) & (~edges.s.isin(base_node_ids))].rename(
                     columns={'s': 'o', 'o': 's'}).assign(p_reverse='1'), ]).drop_duplicates(subset=['s', 'p', 'o'])
            edges_base_to_secondary['o_count'] = edges_base_to_secondary['o'].map(entity_apparitions.to_dict().get)
            edges_base_to_secondary['o_rank'] = edges_base_to_secondary.groupby('s')['o_count'].rank('first',
                                                                                                     ascending=False)
            edges_base_to_secondary['s_score'] = edges_base_to_secondary.s.map(entity_scores.to_dict().get)

            detected_entity_max_neighbours = entity_scores[detected_entities].sort_values(
                ascending=False) * MAX_NEIGHBOURS
            detected_entity_max_neighbours = detected_entity_max_neighbours.astype(int)

            nr_neighbors_to_remove = detected_entity_max_neighbours.sum() - MAX_NODES

            edges_base_to_secondary = edges_base_to_secondary[
                edges_base_to_secondary.o_rank < edges_base_to_secondary.s.map(
                    detected_entity_max_neighbours.to_dict().get)]

            secondary_scores = (
                edges_base_to_secondary[['o', 's_score']].sort_values(by=['s_score'], ascending=False).drop_duplicates(
                    subset='o'))
            secondary_scores = pd.Series(secondary_scores.s_score.values, index=secondary_scores.o)

            edges_base_to_secondary[edges_base_to_secondary.p_reverse == '1'] = swap_columns(
                edges_base_to_secondary[edges_base_to_secondary.p_reverse == '1'], 's', 'o')

            edges = pd.concat([edges[edges.s.isin(base_node_ids) & edges.o.isin(base_node_ids)],
                               edges_base_to_secondary[['s', 'p', 'o']]], ignore_index=True)
            # del edges_base_to_secondary
            entities = pd.concat([base_node_ids, edges.s, edges.o]).drop_duplicates(ignore_index=True)

            entity_scores = pd.concat([entity_scores, secondary_scores, pd.Series({'Q0': 1.0})])
            entity_scores = entities.map(entity_scores.to_dict().get)

            #     secondary_nodes = entity_aparitions[~entity_aparitions.index.isin(base_node_ids)].sort_values(
            #         ascending=False)[:nr_secondary_nodes].index.to_series()
            #     entities = pd.concat([base_node_ids, secondary_nodes]).reset_index(drop=True)
            #     del secondary_nodes
            # else:
            #     entities = base_node_ids

            del entity_apparitions
            # del entity_scores del secondary_nodes

            # add interaction node edges
            inter_edges = pd.DataFrame({'s': 'Q0', 'p': 'P0', 'o': base_node_ids[1:]})
            edges = pd.concat([inter_edges, edges])

            edge_ids = edges['p'].to_list()
            edges.drop(columns=['p'], inplace=True)

            nr_edges = edges.shape[0]
            nr_nodes = entities.shape[0]

            if nr_edges == 0:
                self._log_iteration(row['id'], row['title'], row['label'], skipped=True, has_content=has_content,
                                    title_annotations=title_annotations, content_annotations=content_annotations)
                continue

            print(
                f"[{thread_id}][{datetime.now().time()}] Processed the entities (nodes: {nr_nodes}, edges: {nr_edges})\n")

            replace_dict = pd.Series(entities.index.values, index=entities).to_dict()

            edges = edges.map(replace_dict.get)

            edge_index = torch.tensor(edges.values, dtype=torch.long).t().contiguous()

            print(f"[{thread_id}][{datetime.now().time()}] Processed the edges\n")

            # node_ids = entities.str[1:].astype(int)
            # # log.write(f"Nodes: {node_ids}\n")
            # node_feats = torch.tensor(node_ids, dtype=torch.float).reshape([-1, 1])

            node_labels = do_in_chunks_list(get_labels_from_wikidata_ids, entities, 450,
                                            desc=f"[{thread_id}] Downloading node labels")
            print(
                f"[{thread_id}][{datetime.now().time()}] Downloaded the labels (nodes: {nr_nodes}, edges: {nr_edges})\n")

            node_scores = torch.tensor(entity_scores.values, dtype=torch.float32).reshape((-1, 1))
            node_features = torch.cat([node_scores, self._encode_in_chunks(node_labels)], dim=1)

            print(f"[{thread_id}][{datetime.now().time()}] Processed the nodes \n")

            data = Data(x=node_features, edge_index=edge_index,  # edge_attr = edge_feats,
                        y=row['label'], node_ids=entities.to_list(),
                        edge_ids=edge_ids)  # , content=content_text) title=row['title']

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if data is not None:
                torch.save(data, osp.join(self.processed_dir, f'data_{row.id}.pt'))
            else:
                raise "Data is None"

            print(f"[{thread_id}][{datetime.now().time()}] Data saved \n")

            self._log_iteration(row['id'], row['title'], row['label'], has_content=has_content, count_nodes=nr_nodes,
                                count_edges=nr_edges, title_annotations=title_annotations,
                                content_annotations=content_annotations, wikidata_ids=wikidata_ids,
                                value_counts_before_trim=value_counts_before_trim.to_dict())

            print(f"[{thread_id}][{datetime.now().time()}] Log written \n")

    def _log_iteration(self, data_id, title, label, skipped=False, count_nodes=0, count_edges=0, title_annotations=None,
                       content_annotations=None, wikidata_ids=None, has_content=None, value_counts_before_trim=None):
        if title_annotations is None:
            title_annotations = pd.Series([])
        if content_annotations is None:
            content_annotations = pd.Series([])
        if wikidata_ids is None:
            wikidata_ids = []

        log = pd.DataFrame([[data_id, title, label, skipped, has_content, title_annotations.size,
                             content_annotations.size, str(value_counts_before_trim), count_nodes, count_edges,
                             # str(list(map(str, title_annotations))), str(list(map(str, content_annotations))),
                             str(wikidata_ids)]], columns=LOG_COLUMNS)

        with mutex:
            log.to_csv(self.log_path, mode='a', index=False, header=False)

    def _encode(self, texts, tokenizer, encoder):
        tokens_batch = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=10).to(
            self.device)  # padding='max_length', truncation=True, max_length=5
        outputs_list = encoder(**tokens_batch, output_hidden_states=True)
        embeddings = outputs_list.last_hidden_state[:, 0].detach().cpu()

        # with torch.no_grad():
        #     outputs = model(tokens_tensor, segments_tensors)
        #     hidden_states = outputs[2]

        # input_data = [tokenizer(text, return_tensors="pt").to(self.device) for text in texts]
        #
        # cls_embeddings = [outputs.hidden_states[0][0][0] for outputs in outputs_list]
        #
        # encodings = torch.stack(cls_embeddings).detach().cpu()

        # del tokens_batch, outputs_list
        # clean_gpu(self.device)
        return embeddings

    def _encode_in_chunks(self, texts):
        with gpu_lock:
            # print("enters")
            tokenizer = copy.deepcopy(self.tokenizer)
            encoder = copy.deepcopy(self.encoder).to(self.device)

            # embeddings = do_in_chunks_tensor(self._encode, texts, 3770, tokenizer, encoder) #1650
            embeddings = self._encode(texts, tokenizer, encoder)

            del encoder
            # clean_gpu(self.device)
            # print("leaves")
            return embeddings

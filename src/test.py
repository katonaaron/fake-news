import json
import os
from datetime import datetime

import pandas as pd
import tagme
from tqdm import tqdm

from src.core.kg_construct import construct_graph
from src.wikidata import get_wikidata_ids_from_uris

RHO_TITLE = 0.08


# RHO_TEXT_SENTENCE = 0.1
# from franz.openrdf.connect import ag_connect
# from franz.openrdf.sail import AllegroGraphServer
#
# from util import setup_env_var

# setup_env_var('AGRAPH_HOST', 'http://ag1qzxuwy43npnmy.allegrograph.cloud/', 'Hostname')
# setup_env_var('AGRAPH_PORT', '443', 'Port')
# setup_env_var('AGRAPH_USER', 'admin', 'Username')
# setup_env_var('AGRAPH_PASSWORD', 'n8KKbtEOFkHOiGnRx8msou', 'Password')

# setup_env_var('AGRAPH_HOST', 'http://localhost', 'Hostname')
# setup_env_var('AGRAPH_PORT', '10035', 'Port')
# setup_env_var('AGRAPH_USER', 'user', 'Username')
# setup_env_var('AGRAPH_PASSWORD', 'password', 'Password')
#
# AGRAPH_HOST = os.environ.get('AGRAPH_HOST', 'http://ag1qzxuwy43npnmy.allegrograph.cloud/')
# AGRAPH_PORT = int(os.environ.get('AGRAPH_PORT', '80'))
# AGRAPH_USER = os.environ.get('AGRAPH_USER', 'admin')
# AGRAPH_PASSWORD = os.environ.get('AGRAPH_PASSWORD', 'n8KKbtEOFkHOiGnRx8msou')
#
# print("Connecting to AllegroGraph server --",
#       "host:'%s' port:%s" % (AGRAPH_HOST, AGRAPH_PORT))
# server = AllegroGraphServer(AGRAPH_HOST, AGRAPH_PORT,
#                             AGRAPH_USER, AGRAPH_PASSWORD)

# with ag_connect(f'test-{1}', create=True, clear=True) as conn:
#     ut = conn.namespace('http://utcluj.ro/fake-news/')
#     wd = conn.namespace('http://www.wikidata.org/entity/')
#     wdt = conn.namespace('http://www.wikidata.org/prop/direct/')

class Edge:
    def __init__(self, wdt_id, out_nodes, edge_label=None):
        self.wd_id = wdt_id
        self.label = edge_label
        self.out_nodes = out_nodes


class Node:

    def __init__(self, wd_id, label=None, content=None):
        self.wd_id = wd_id
        self.label = label
        self.content = content
        self.out_edges = []

    def add_edges(self, children, edge_id, edge_label=None):
        self.out_edges.append(Edge(edge_id, edge_label, children))


class DocumentGroup:

    def __init__(self, wd_id, title, documents):
        self.wd_id = wd_id
        self.title = title
        self.documents = documents


class Document:

    def __init__(self, wd_id, title, content):
        self.wd_id = wd_id
        self.title = title
        self.content = content


class Temp:
    def _process_chunk(self, chunk, thread_id):
        tagme.GCUBE_TOKEN = "f228d32f-ce3a-4c61-b8ce-da0a3f2e154d-843339462"

        total_rows = chunk.shape[0]
        for index, instance in tqdm(chunk.iterrows(), total=total_rows, desc="Processing chunk #{}".format(thread_id)):
            if instance.label == 'half-true':
                print(
                    f"[{thread_id}][{datetime.now().time()}] Skipped {instance.event_id}\n")
                continue



            # claim_entity_nodes = self.detect_entity_nodes(instance.claim)
            # claim_node = Node('Q01', '<claim>', instance.claim)
            # claim_node.add_edges(claim_entity_nodes, 'P01', edge_label='<in_claim>')

            reports = pd.json_normalize(instance.reports)
            report_nodes = []
            report_docs = []
            for _, report in reports.iterrows():
                tokenized = pd.json_normalize(report.tokenized)

                sentences = [Document(f'Q02_{report.report_id}_{sent_idx}', sent, sent) for sent_idx, sent in tokenized.sent.items()]

                report_docs.append(DocumentGroup(f'Q02_{report.report_id}', f'<report-{report.report_id}>', sentences))
                #
                # sent_nodes = []
                # for sent_idx, sent in tokenized.sent.items():
                #     sent_entity_nodes = self.detect_entity_nodes(sent)
                #     sent_node = Node(f'Q03_{report.report_id}_{sent_idx}', sent)
                #     sent_node.add_edges(sent_entity_nodes, 'P03', edge_label='<in_sentence>')
                #     sent_nodes.append(sent_node)
                #
                # report_node = Node(f'Q02_{report.report_id}', report.report_id, report.content)
                # report_node.add_edges(sent_nodes, 'P02', edge_label='<in_report>')
                # report_nodes.append(report_node)

            # interaction_node = Node('Q00', '<interaction>', content='')
            # interaction_node.add_edges([claim_node.wd_id, *report_nodes], 'P00', '<has>')

            documentGraph = DocumentGroup('Q0', '<interaction>', [
                Document('Q01', '<claim>', instance.claim),
                DocumentGroup('Q02', '<reports>', report_docs)
            ])

            graph = construct_graph(documentGraph)

            # # content_annotations = []
            # # has_content = False
            # # if self.news_content:
            # #     if instance['label'] == 1:
            # #         label = 'real'
            # #     else:
            # #         label = 'fake'
            # #
            # #     content_path = osp.join(self.raw_dir, self.dataset, label, instance['id'], 'news content.json')
            # #     if osp.isfile(content_path):
            # #
            # #         with open(content_path) as f:
            # #             news_content = json.load(f)
            # #             if news_content['text']:
            # #                 has_content = True
            # #                 content_text = news_content['text']
            # #                 try:
            # #                     content_annotations = list(
            # #                         tagme.annotate(content_text, long_text=TEXT_ANNOTATE_WINDOW).get_annotations(
            # #                             RHO_TEXT))
            # #                     if not content_annotations:
            # #                         content_annotations = []
            # #                     if len(content_annotations) > 0:
            # #                         content_annotations.sort(key=lambda ann: ann.score, reverse=True)
            # #                     # if len(content_annotations) > TEXT_MAX_ANNOTATIONS:
            # #                     #     content_annotations = content_annotations[:TEXT_MAX_ANNOTATIONS]
            # #                 except Exception as e:
            # #                     print(e, content_text)
            #
            # # log.write(
            # #     f"Annotations ({len(annotations)}): {[ann.entity_title + ' (' + str(ann.score) + ')' for ann in annotations]}\n")
            #
            # # log.write(f"Uris: {uris}\n")
            #
            # # if len(annotations) == 0:
            # #     # log.write(f"No annotations found\n")
            # #     # log.write("\n")
            # #     # log.flush()
            # #     self._log_iteration(instance['id'], instance['title'], instance['label'], skipped=True)
            # #     continue
            #
            # # log.write(f"Wikidata ids: {wikidata_ids}\n")
            #
            # if len(wikidata_ids) == 0:
            #     self._log_iteration(instance['id'], instance['title'], instance['label'], skipped=True,
            #                         has_content=has_content,
            #                         title_annotations=title_annotations, content_annotations=content_annotations)
            #     continue
            #
            # edges = self.all_triples[(self.all_triples.s.isin(wikidata_ids)) | (self.all_triples.o.isin(wikidata_ids))]
            #
            # # log.write(f"Edges: {edges.shape[0]}\n")
            # # log.write("\n")
            # # log.flush()
            #
            # # nr_edges = edges.shape[0]
            # # if nr_edges == 0:
            # #     self._log_iteration(row['id'], row['title'], row['label'], skipped=True,
            # #                         title_annotations=title_annotations, content_annotations=content_annotations)
            # #     continue
            # #
            #
            # base_node_ids = pd.Series(['Q0'] + list(wikidata_ids))
            #
            # print(
            #     f"[{thread_id}][{datetime.now().time()}] Start processing entities (base nodes: {base_node_ids.shape[0]})\n")
            #
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
            #
            # # add interaction node edges
            # inter_edges = pd.DataFrame({'s': 'Q0', 'p': 'P0', 'o': base_node_ids[1:]})
            # edges = pd.concat([inter_edges, edges])
            #
            # edge_ids = edges['p'].to_list()
            # edges.drop(columns=['p'], inplace=True)
            #
            # nr_edges = edges.shape[0]
            # nr_nodes = entities.shape[0]
            #
            # if nr_edges == 0:
            #     self._log_iteration(instance['id'], instance['title'], instance['label'], skipped=True,
            #                         has_content=has_content,
            #                         title_annotations=title_annotations, content_annotations=content_annotations)
            #     continue
            #
            # print(
            #     f"[{thread_id}][{datetime.now().time()}] Processed the entities (nodes: {nr_nodes}, edges: {nr_edges})\n")
            #
            # replace_dict = pd.Series(entities.index.values, index=entities).to_dict()
            #
            # edges = edges.map(replace_dict.get)
            #
            # edge_index = torch.tensor(edges.values, dtype=torch.long).t().contiguous()
            #
            # print(f"[{thread_id}][{datetime.now().time()}] Processed the edges\n")
            #
            # # node_ids = entities.str[1:].astype(int)
            # # # log.write(f"Nodes: {node_ids}\n")
            # # node_feats = torch.tensor(node_ids, dtype=torch.float).reshape([-1, 1])
            #
            # node_labels = do_in_chunks_list(get_labels_from_wikidata_ids, entities, 450,
            #                                 desc=f"[{thread_id}] Downloading node labels")
            # print(
            #     f"[{thread_id}][{datetime.now().time()}] Downloaded the labels (nodes: {nr_nodes}, edges: {nr_edges})\n")
            #
            # node_features = self._encode_in_chunks(node_labels)
            #
            # print(f"[{thread_id}][{datetime.now().time()}] Processed the nodes \n")
            #
            # data = Data(x=node_features, edge_index=edge_index,  # edge_attr = edge_feats,
            #             y=instance['label'], node_ids=entities.to_list(),
            #             edge_ids=edge_ids)  # , content=content_text) title=row['title']
            #
            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue
            #
            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)
            #
            # if data is not None:
            #     torch.save(data, osp.join(self.processed_dir, f'data_{instance.id}.pt'))
            # else:
            #     raise "Data is None"
            #
            # print(f"[{thread_id}][{datetime.now().time()}] Data saved \n")
            #
            # self._log_iteration(instance['id'], instance['title'], instance['label'], has_content=has_content,
            #                     count_nodes=nr_nodes,
            #                     count_edges=nr_edges,
            #                     title_annotations=title_annotations, content_annotations=content_annotations,
            #                     wikidata_ids=wikidata_ids, value_counts_before_trim=value_counts_before_trim.to_dict())
            #
            # print(f"[{thread_id}][{datetime.now().time()}] Log written \n")

    def detect_entities(self, text: str) -> set[str]:
        annotations = self.annotate_short_text(text)

        if len(annotations) == 0:
            return set()

        uris = set([ann.uri() for ann in annotations])

        wikidata_ids = set(get_wikidata_ids_from_uris(uris))
        if None in wikidata_ids:
            wikidata_ids.remove(None)
        return wikidata_ids

    def detect_entity_nodes(self, text: str) -> list[Node]:
        entities = self.detect_entities(text)
        return [Node(e) for e in entities]

    def detect_entities_merged(self, texts: list[str]) -> set[str]:
        annotations = []
        uris = set()
        for text in texts:
            anns = self.annotate_short_text(text)
            if len(anns) == 0:
                continue
            annotations.extend(anns)
            uris.update(set([ann.uri() for ann in anns]))

        if len(uris) == 0:
            return set()

        wikidata_ids = set(get_wikidata_ids_from_uris(uris))
        if None in wikidata_ids:
            wikidata_ids.remove(None)
        return wikidata_ids

    def annotate_short_text(self, text, rho=RHO_TITLE):
        return list(tagme.annotate(text).get_annotations(rho))


with open(os.path.join('data', 'raw', 'LIAR-RAW', f"train.json"), encoding="utf8") as data_file:
    data = json.load(data_file)

df = pd.json_normalize(data)
# reports = pd.json_normalize(df.reports.iloc[0])
# tokenized = pd.json_normalize(reports.tokenized.iloc[0])
# tokenized


temp = Temp()
temp._process_chunk(df, 0)

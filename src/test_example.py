import pandas as pd
import tagme
import os.path as osp

import torch
from torch_geometric.data import Data

from src.process_ds import encode_short, add_text_embedding, tokenizer, encoder
from src.util import do_in_chunks_list
from src.wikidata import get_wikidata_ids_from_uris, get_labels_from_wikidata_ids

tagme.GCUBE_TOKEN = "f228d32f-ce3a-4c61-b8ce-da0a3f2e154d-843339462"

if __name__ == '__main__':
    text = "Leonardo DiCaprio Tells President Obama That He's Going to Mars"
    label = 1  #fake
    test_id = '	politifact14270Title'

    title_annotations = list(tagme.annotate(text).get_annotations(0.08))

    print(str(list(map(str, title_annotations))))

    uris = set([ann.uri() for ann in title_annotations])

    wikidata_ids = get_wikidata_ids_from_uris(uris)

    print(wikidata_ids)

    wikidata_ids = set(wikidata_ids)

    edges = pd.read_csv(osp.join('data', 'raw', 'wikidata5m', 'wikidata5m_all_triplet.txt'), names=['s', 'p', 'o'],
                        delimiter='\t')
    edges = edges[(edges.s.isin(wikidata_ids)) | (edges.o.isin(wikidata_ids))]

    entity_aparitions = pd.concat([edges.s, edges.o]).value_counts()
    value_counts_before_trim = entity_aparitions.value_counts().sort_values(ascending=False)

    base_node_ids = pd.Series(['Q0'] + list(wikidata_ids))

    nr_secondary_nodes = min(entity_aparitions.shape[0], 1000) - base_node_ids.shape[0]
    if nr_secondary_nodes > 0:
        secondary_nodes = entity_aparitions[~entity_aparitions.index.isin(base_node_ids)].sort_values(
            ascending=False)[:nr_secondary_nodes].index.to_series()
        entities = pd.concat([base_node_ids, secondary_nodes]).drop_duplicates().reset_index(drop=True)
        del secondary_nodes
    else:
        entities = base_node_ids

    del entity_aparitions

    edges = edges[edges.s.isin(entities) & edges.o.isin(entities)]

    # add interaction node edges
    inter_edges = pd.DataFrame({'s': 'Q0', 'p': 'P0', 'o': base_node_ids[1:]})
    edges = pd.concat([inter_edges, edges])

    edge_ids = edges['p'].to_list()
    edges.drop(columns=['p'], inplace=True)

    nr_edges = edges.shape[0]
    nr_nodes = entities.shape[0]

    if nr_edges == 0:
        raise "0"

    replace_dict = pd.Series(entities.index.values, index=entities).to_dict()

    edges = edges.map(replace_dict.get)

    edge_index = torch.tensor(edges.values, dtype=torch.long).t().contiguous()

    # node_ids = entities.str[1:].astype(int)
    # # log.write(f"Nodes: {node_ids}\n")
    # node_feats = torch.tensor(node_ids, dtype=torch.float).reshape([-1, 1])

    node_labels = do_in_chunks_list(get_labels_from_wikidata_ids, entities, 450,
                                    desc=f"Downloading node labels")

    node_features = encode_short(node_labels, tokenizer, encoder)

    data = Data(x=node_features, edge_index=edge_index,  # edge_attr = edge_feats,
                y=label, node_ids=entities.to_list(),
                edge_ids=edge_ids)  # , content=content_text) title=row['title']

    add_text_embedding(data, test_id)

    if data is not None:
        torch.save(data, osp.join('data', 'processed', f'data_{test_id}.pt'))
    else:
        raise "Data is None"
import argparse
import sys

import graph_force
import pandas as pd
import os.path as osp
import torch
from torch.nn import DataParallel
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, fidelity, unfaithfulness
from transformers import BertTokenizer, BertModel

from src.model import Model
from src.util import tagme_annotate, swap_columns, do_in_chunks_list, do_in_chunks_dict, truncate
from src.wikidata import get_wikidata_id_map_from_uris, get_labels_from_wikidata_ids, get_labels_dict_from_wikidata_ids, \
    get_property_labels_dict_from_wikidata_ids

RHO_TITLE = 0.08
RHO_TEXT = 0.1
TEXT_ANNOTATE_WINDOW = 1  # 3
TAGME_MAX_TEXT_LENGTH = 700
MAX_NEIGHBOURS = 7  # 5

device = 'cuda:0'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoder = BertModel.from_pretrained('bert-base-uncased').to(device)
encoder.eval()

filename_all_triples = 'wikidata5m_all_triplet.txt'
all_triples = pd.read_csv(osp.join('data', 'raw', 'wikidata5m', filename_all_triples),
                               names=['s', 'p', 'o'], delimiter='\t')


def construct_graph(title, content, label=None):
    response = tagme_annotate(title)
    title_annotations = list(response.get_annotations(RHO_TITLE))

    content_annotations = []
    if content:
        remaining_text = content
        while remaining_text:
            current_text = remaining_text
            if len(current_text) > TAGME_MAX_TEXT_LENGTH:
                index = current_text.rfind(' ', 0, TAGME_MAX_TEXT_LENGTH - 1)
                if index < 0:
                    print("cannot split content ", file=sys.stderr)
                    continue
                else:
                    remaining_text = current_text[index:]
                    current_text = current_text[:index]
            else:
                remaining_text = None

            response = tagme_annotate(current_text, long_text=TEXT_ANNOTATE_WINDOW)
            content_annotations += list(response.get_annotations(RHO_TEXT))

    # print(f"[{thread_id}][{datetime.now().time()}] {row['id']} annotated\n")

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

    # annotations.to_csv(annotations_path, mode='w', index=False)

    uris = set(annotations['uri'])
    # log.write(f"Uris: {uris}\n")

    if len(uris) == 0:
        # log.write(f"No annotations found\n")
        # log.write("\n")
        # log.flush()
        # self._log_iteration(row['id'], row['title'], row['label'], skipped=True, has_content=has_content)
        return None

    wikidata_uri_to_id = get_wikidata_id_map_from_uris(uris)

    wikidata_ids = set(wikidata_uri_to_id.values())
    if None in wikidata_ids:
        wikidata_ids.remove(None)

    if len(wikidata_ids) == 0:
        # self._log_iteration(row['id'], row['title'], row['label'], skipped=True, has_content=has_content,
        #                     title_annotations=title_annotations, content_annotations=content_annotations)
        return None

    # del annotations[~annotations.uri.isin(wikidata_uri_to_id.keys())]

    edges = all_triples[(all_triples.s.isin(wikidata_ids)) | (all_triples.o.isin(wikidata_ids))]


    detected_entities = pd.Series(list(wikidata_ids))
    base_node_ids = pd.concat([pd.Series(['Q0']), detected_entities])

    # print(
    #     f"[{thread_id}][{datetime.now().time()}] Start processing entities (base nodes: {base_node_ids.shape[0]})\n")

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

    # print(
    #     f"[{thread_id}][{datetime.now().time()}] Processed the entities (nodes: {nr_nodes}, edges: {nr_edges})\n")

    replace_dict = pd.Series(entities.index.values, index=entities).to_dict()

    edges = edges.map(replace_dict.get)

    edge_index = torch.tensor(edges.values, dtype=torch.long).t().contiguous()

    # print(f"[{thread_id}][{datetime.now().time()}] Processed the edges\n")

    # node_ids = entities.str[1:].astype(int)
    # # log.write(f"Nodes: {node_ids}\n")
    # node_feats = torch.tensor(node_ids, dtype=torch.float).reshape([-1, 1])

    node_labels = do_in_chunks_list(get_labels_from_wikidata_ids, entities, 450,
                                    desc=f"Downloading node labels")
    # print(
    #     f"[{thread_id}][{datetime.now().time()}] Downloaded the labels (nodes: {nr_nodes}, edges: {nr_edges})\n")

    node_scores = torch.tensor(entity_scores.values, dtype=torch.float32).reshape((-1, 1))
    label_embeddings = encode_labels(node_labels[1:])
    if content:
        content_embedding = encode_content(content)
    else:
        content_embedding = torch.zeros(label_embeddings.shape[1])

    node_features = torch.cat([node_scores, torch.cat([content_embedding.reshape(1,-1), label_embeddings])], dim=1)



    # print(f"[{thread_id}][{datetime.now().time()}] Processed the nodes \n")

    return Data(x=node_features, edge_index=edge_index,  # edge_attr = edge_feats,
                y=label, node_ids=entities.to_list(),
                edge_ids=edge_ids)  # , content=content_text) title=row['title']


def encode_labels(labels):
    tokens_batch = tokenizer(labels, return_tensors="pt", padding='max_length', truncation=True, max_length=10).to(
        device)  # padding='max_length', truncation=True, max_length=5
    outputs_list = encoder(**tokens_batch, output_hidden_states=True)
    embeddings = outputs_list.last_hidden_state[:, 0].detach().cpu()
    return embeddings

def encode_content(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        device)  # padding='max_length', truncation=True, max_length=5
    outputs = encoder(**tokens, output_hidden_states=True)
    embedding = outputs.last_hidden_state[:, 0].detach().cpu()
    return embedding[0]

def explain(model, data, test_id):
    data = data.to(device)
    with torch.no_grad():
        label = data.y
        prediction = model(data.x, data.edge_index).sigmoid().item()
        print(  # f"Expected label: {data.y}, Predicted: {F.softmax(model(data.x, data.edge_index))}")
            f"Expected label: {label}, Predicted: {prediction}")

    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=200), explanation_type='model',
                          node_mask_type='object', edge_mask_type='object',
                          model_config=ModelConfig(mode='binary_classification', task_level='graph',
                                                   return_type='raw', ), )

    explanation = explainer(data.x, data.edge_index)
    print(f'Generated explanations in {explanation.available_explanations}')

    fidPlus, fidMinus = fidelity(explainer, explanation)
    unf = unfaithfulness(explainer, explanation)
    print(f'Fid Plus: {fidPlus}, FidMinus: {fidMinus}, Unfaithfulness: {unf}')

    # test_log = pd.DataFrame(columns=['id', 'fake', 'prediction', 'fid+', 'fid-', 'unfaithfulness'], data=[
    #     [test_id, label, prediction, fidPlus, fidMinus, unf]
    # ])
    # test_log.to_csv(test_log_path, index=False, header=False, mode='a')

    # path = 'feature_importance.png'
    # explanation.visualize_feature_importance(path, top_k=10)
    # print(f"Feature importance plot has been saved to '{path}'")
    # visualize_feature_importance(explanation, data)

    # explanation_path = osp.join('explanations', test_id)
    # Path(explanation_path).mkdir(parents=True, exist_ok=True)

    nodes = pd.DataFrame(
        {'id': data.node_ids, 'importance': explanation.node_mask.cpu().numpy().reshape(-1)})

    node_label_dict = do_in_chunks_dict(get_labels_dict_from_wikidata_ids, set(nodes.id), 500)
    nodes['label'] = nodes.id.map(node_label_dict.get)

    # nodes.to_csv(osp.join(explanation_path, f"node_importance-{test_id}.csv"), columns=['label', 'importance'],
    #              index=False)

    # path = osp.join(explanation_path, f'subgraph-{test_id}.pdf')
    # explanation.visualize_graph(path, node_labels=nodes.label.to_list())
    # print(f"Subgraph visualization plot has been saved to '{path}'")

    edges = pd.DataFrame(data.edge_index.cpu().numpy()).T
    edges.columns = ["s", "o"]
    edges["s"] = edges["s"].map(lambda idx: nodes.id[idx])
    edges["p"] = data.edge_ids
    edges["o"] = edges["o"].map(lambda idx: nodes.id[idx])

    edge_label_dict = do_in_chunks_dict(get_property_labels_dict_from_wikidata_ids, set(edges.p), 500)

    edges["from"] = edges["s"].map(node_label_dict.get)
    edges["prop"] = edges["p"].map(edge_label_dict.get)
    edges["to"] = edges["o"].map(node_label_dict.get)

    edges["importance"] = explanation.edge_mask.cpu().numpy().reshape(-1)

    # edges.to_csv(osp.join(explanation_path, f"edge_importance-{test_id}.csv"),
    #              columns=['from', 'prop', 'to', 'importance'], index=False)

    # nodes = nodes[nodes.importance > 0.0]
    # edges = edges[edges.importance > 0.0]

    # with open(osp.join('report_template', 'report_template_begin.html')) as f:
    #     report_begin = f.read()
    #
    # with open(osp.join('report_template', 'report_template_end.html')) as f:
    #     report_end = f.read()

    # graph = ''

    # log_row = log[log['id'] == test_id]

    out = dict()
    out['id'] = test_id
    # out['title'] = log_row['title'].values[0]
    out['label'] = label
    # out['title_annotations'] = log_row['title_annotations'].values[0]
    # out['content_annotations'] = log_row['content_annotations'].values[0]

    out['fidPlus'] = fidPlus
    out['fidMinus'] = fidMinus
    out['unfaithfulness'] = unf

    if prediction <= 0.5:
        out['confidence'] = truncate(1 - prediction, 2)
        out['prediction'] = 0
    else:
        out['confidence'] = truncate(prediction, 2)
        out['prediction'] = 1

    if label == 0:
        label_name = 'real'
    else:
        label_name = 'fake'

    # content_path = os.path.join(content_dir, label_name, test_id, 'news content.json')
    # with open(content_path) as f:
    #     news_content = json.load(f)
    #     content_text = news_content['text']
    #     out['content'] = content_text

    max_node_size = 32
    max_edge_size = 12

    edge_pairs = data.edge_index.cpu().numpy().astype(int).transpose().tolist()
    edge_pairs = list(map(tuple, edge_pairs))
    pos = graph_force.layout_from_edge_list(nodes.shape[0], edge_pairs)
    # adjacency_matrix = to_dense_adj(data.edge_index)[0].cpu().numpy().astype(int)

    # adjacency_matrix = logical_or(adjacency_matrix.transpose(), adjacency_matrix).astype(int)
    # print("Computing node positions")
    # pos = forceatlas2(adjacency_matrix)
    # print("Computation done")
    out_nodes = []
    for node_idx, row in nodes.iterrows():
        node = row.id
        importance = row.importance
        label = row.label + f' ({truncate(importance, 2)})'

        if importance == 0:
            continue

        node_size = max(int(max_node_size * importance), 1)
        x = pos[node_idx][0]
        y = pos[node_idx][1]
        # x = y = 0
        out_nodes.append({
            'id': node,
            'label': label,
            'size': node_size,
            'x': x,
            'y': y
        })
    out['nodes'] = out_nodes
    # graph += f'graph.addNode("{node}", {{size: {node_size}, x:{x}, y:{y}, label: "{label} ({"%.2f" % importance})", color: BLUE}});\n'

    out_edges = []
    for _, row in edges.iterrows():
        s = row.s
        p = row.p
        o = row.o
        importance = row.importance
        if p in edge_label_dict:
            edge_label = edge_label_dict[p]
        else:
            edge_label = p
        edge_label = edge_label + f' ({truncate(importance, 2)})'

        if importance == 0:
            continue

        edge_size = max(int(max_edge_size * importance), 1)

        # graph += f'graph.addEdge("{s}", "{o}", {{type: "arrow", label: "{edge_label} ({"%.2f" % importance})", size: {edge_size}}});\n'
        out_edges.append({
            's': s,
            'o': o,
            'label': edge_label,
            'size': edge_size
        })
    out['edges'] = out_edges
    return out

def read_model():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--device', type=str, default=device, help='specify cuda devices')

    # hyper-parameters
    parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=30, help='maximum number of epochs')
    parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
    parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')
    parser.add_argument('--model', type=str, default='sage', help='model type, [gcn, gat, sage]')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


    # if epoch:
    #     modelPath = f'model/model-{epoch}.pt'
    # else:
    model_path = f'model/model.pt'
    state_dict = torch.load(model_path)

    args.num_classes = 1
    args.num_features = state_dict['conv1.lin_l.weight'].shape[1]


    model = Model(args, concat=args.concat)

    if args.multi_gpu:
        model = DataParallel(model)
    model.load_state_dict(state_dict)
    model.eval()

    return model.to(args.device)
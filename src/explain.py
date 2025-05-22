import argparse
import json
import os
import os.path as osp
from pathlib import Path

import graph_force
import pandas as pd
import torch
from torch.nn import DataParallel
from torch.utils.data import random_split
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, fidelity, unfaithfulness
from torch_geometric.loader import DataLoader, DataListLoader
from tqdm import tqdm

from dataset.fnn_dataset import FNNWikidata5MDataset
from model import Model
from src.util import truncate
from util import do_in_chunks_dict
from wikidata import get_labels_dict_from_wikidata_ids, get_property_labels_dict_from_wikidata_ids

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, dataset)
# data = dataset[0]

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

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

dataset = FNNWikidata5MDataset(root='data', dataset=args.dataset, device=args.device, skip_processing=True,
                               news_content=True)

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

if args.multi_gpu:
    loader = DataListLoader
else:
    loader = DataLoader

# train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True, exclude_keys=['text', 'content'])

raw_dir = os.path.join('data', 'raw')
processed_dir = os.path.join('data', 'processed')
res_dir = os.path.join(processed_dir, 'res')
log_path = os.path.join(processed_dir, f"politifact_log.csv")
content_dir = os.path.join(raw_dir, 'politifact')
log = pd.read_csv(log_path)

test_log_path = osp.join('explanations', 'test_log.csv')


def explain(model, data, test_id):
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

    test_log = pd.DataFrame(columns=['id', 'fake', 'prediction', 'fid+', 'fid-', 'unfaithfulness'], data=[
        [test_id, label, prediction, fidPlus, fidMinus, unf]
    ])
    test_log.to_csv(test_log_path, index=False, header=False, mode='a')

    # path = 'feature_importance.png'
    # explanation.visualize_feature_importance(path, top_k=10)
    # print(f"Feature importance plot has been saved to '{path}'")
    # visualize_feature_importance(explanation, data)

    explanation_path = osp.join('explanations', test_id)
    Path(explanation_path).mkdir(parents=True, exist_ok=True)

    nodes = pd.DataFrame(
        {'id': data.node_ids, 'importance': explanation.node_mask.cpu().numpy().reshape(-1)})

    node_label_dict = do_in_chunks_dict(get_labels_dict_from_wikidata_ids, set(nodes.id), 500)
    nodes['label'] = nodes.id.map(node_label_dict.get)

    nodes.to_csv(osp.join(explanation_path, f"node_importance-{test_id}.csv"), columns=['label', 'importance'],
                 index=False)

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

    edges.to_csv(osp.join(explanation_path, f"edge_importance-{test_id}.csv"),
                 columns=['from', 'prop', 'to', 'importance'], index=False)

    # nodes = nodes[nodes.importance > 0.0]
    # edges = edges[edges.importance > 0.0]

    with open(osp.join('report_template', 'report_template_begin.html')) as f:
        report_begin = f.read()

    with open(osp.join('report_template', 'report_template_end.html')) as f:
        report_end = f.read()

    # graph = ''

    log_row = log[log['id'] == test_id]

    out = dict()
    out['id'] = test_id
    out['title'] = log_row['title'].values[0]
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

    content_path = os.path.join(content_dir, label_name, test_id, 'news content.json')
    with open(content_path) as f:
        news_content = json.load(f)
        content_text = news_content['text']
        out['content'] = content_text

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

    report_path = osp.join(explanation_path, f'report_{test_id}.html')
    with open(report_path, 'w') as f:
        f.write(report_begin + json.dumps(out) + report_end)
    print(f"Report generated to '{report_path}'")

    res_log = pd.DataFrame([[out['id'], out['label'], out['prediction'], out['confidence']]],
                           columns=['id', 'label', 'prediction', 'confidence'])
    res_log_path = osp.join('explanations', 'log.csv')
    if os.path.exists(res_log_path):
        create_header = False
    else:
        create_header = True
    res_log.to_csv(res_log_path, mode='a', index=False, header=create_header)

    # # setup_env_var('AGRAPH_HOST', 'https://ag1qzxuwy43npnmy.allegrograph.cloud/', 'Hostname')
    # # setup_env_var('AGRAPH_PORT', '8443', 'Port')
    # # setup_env_var('AGRAPH_USER', 'admin', 'Username')
    # # setup_env_var('AGRAPH_PASSWORD', 'n8KKbtEOFkHOiGnRx8msou', 'Password')
    #
    # setup_env_var('AGRAPH_HOST', 'http://localhost', 'Hostname')
    # setup_env_var('AGRAPH_PORT', '10035', 'Port')
    # setup_env_var('AGRAPH_USER', 'user', 'Username')
    # setup_env_var('AGRAPH_PASSWORD', 'password', 'Password')
    #
    # AGRAPH_HOST = os.environ.get('AGRAPH_HOST', 'localhost')
    # AGRAPH_PORT = int(os.environ.get('AGRAPH_PORT', '10035'))
    # AGRAPH_USER = os.environ.get('AGRAPH_USER', 'test')
    # AGRAPH_PASSWORD = os.environ.get('AGRAPH_PASSWORD', 'xyzzy')
    #
    # print("Connecting to AllegroGraph server --",
    #       "host:'%s' port:%s" % (AGRAPH_HOST, AGRAPH_PORT))
    # server = AllegroGraphServer(AGRAPH_HOST, AGRAPH_PORT,
    #                             AGRAPH_USER, AGRAPH_PASSWORD)
    #
    # with ag_connect(f'test-{test_id}', create=True, clear=True) as conn:
    #     conn.enableRDFStar()
    #
    #     ut = conn.namespace('http://utcluj.ro/fake-news/')
    #     wd = conn.namespace('http://www.wikidata.org/entity/')
    #     wdt = conn.namespace('http://www.wikidata.org/prop/direct/')
    #
    #     graph = ut[test_id]
    #
    #     for _, row in nodes.iterrows():
    #         node = wd[row.id]
    #         label = conn.createLiteral(row.label)
    #         importance = conn.createLiteral(row.importance)
    #         conn.add(node, RDFS.LABEL, label, contexts=[graph])
    #         conn.add(node, ut.has_importance, importance, contexts=[graph])
    #
    #     for edge_id, edge_label in edge_label_dict.items():
    #         edge = wdt[edge_id]
    #         label = conn.createLiteral(edge_label)
    #         conn.add(edge, RDFS.LABEL, label, contexts=[graph])
    #
    #     for _, row in edges.iterrows():
    #         s = wd[row.s]
    #         p = wdt[row.p]
    #         o = wd[row.o]
    #         importance = row.importance
    #
    #         conn.add(s, p, o, contexts=[graph])
    #         quoted_triple = QuotedTriple(s, p, o)
    #         conn.add(quoted_triple, ut.has_importance, importance, contexts=[graph])


def main():
    # filename_entity = 'wikidata5m_entity.txt'
    # # entities = pd.read_csv(osp.join('data', 'raw', filename_entity), encoding="utf-8", delimiter='\t', usecols=[0, 1],
    # #                        header=None)
    #
    # chunks = []
    # df = pd.read_csv(osp.join('data', 'raw', filename_entity), encoding="utf-8", delimiter='\t', usecols=[0, 1], header=None, chunksize=1000)  # gives TextFileReader
    # for chunk in df:
    #     chunks.append(chunk[chunk[0].isin(labels)])
    # entity_names = pd.concat(chunks)

    # entity_names = dict()
    # with open(osp.join('data', 'raw', filename_entity), 'r', encoding="utf8") as f:
    #     for line in f:
    #         terms = line.strip().split('\t')
    #         entity_id = terms[0]
    #         if entity_id in entity_id_set:
    #             entity_names[entity_id] = terms[1]

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device
    # model = GCN().to(device)
    model = Model(args, concat=args.concat)
    if args.multi_gpu:
        model = DataParallel(model)

    epoch = None
    if epoch:
        modelPath = f'model/model-{epoch}.pt'
    else:
        modelPath = f'model/model.pt'
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    # model.tokenizer = None
    # model.encoder = None

    model = model.to(args.device)

    # dataset = 'Cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    # dataset = Planetoid(path, dataset)
    # data = dataset[0]
    #
    #
    # class GCN(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.conv1 = GCNConv(dataset.num_features, 16)
    #         self.conv2 = GCNConv(16, dataset.num_classes)
    #
    #     def forward(self, x, edge_index):
    #         x = F.relu(self.conv1(x, edge_index))
    #         x = F.dropout(x, training=self.training)
    #         x = self.conv2(x, edge_index)
    #         return F.log_softmax(x, dim=1)
    #
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN().to(device)
    # data = data.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #
    # for epoch in range(1, 201):
    #     model.train()
    #     optimizer.zero_grad()
    #     out = model(data.x, data.edge_index)
    #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #     loss.backward()
    #     optimizer.step()
    #
    # explainer = Explainer(
    #     model=model,
    #     algorithm=GNNExplainer(epochs=200),
    #     explanation_type='model',
    #     node_mask_type='attributes',
    #     edge_mask_type='object',
    #     model_config=dict(
    #         mode='multiclass_classification',
    #         task_level='graph',
    #         return_type='log_probs',
    #     ),
    # )

    # explainer = Explainer(
    #     model=model,
    #     algorithm=PGExplainer(epochs=30, lr=0.003),
    #     explanation_type='model',
    #     node_mask_type='attributes',
    #     edge_mask_type='object',
    #     model_config=ModelConfig(
    #         mode='binary_classification',
    #         task_level='graph',
    #         return_type='probs',
    #     ),
    # )
    #
    # explainer = explainer.algorithm.to(args.device)
    #
    # for epoch in tqdm(range(30)):
    #     for i, data in enumerate(train_loader):
    #         data = data.to(device)
    #         loss = explainer.algorithm.train(epoch, model, data.x, data.edge_index, target=data.target, index=0)

    # test_no = 1
    # data = test_set[test_no]

    test_log = pd.DataFrame(columns=['id', 'fake','prediction', 'fid+', 'fid-','unfaithfulness'])
    if not osp.isfile(test_log_path):
        test_log.to_csv(test_log_path, index=False)

    # for test_no, data in tqdm(enumerate(test_loader))
    for test_no, data in tqdm(enumerate(test_set), total=num_test):
        data = data.to(device)

        test_id = dataset.processed_file_names[test_set.indices[test_no]][5:-3]
        print(f"Explaining test {test_id}")

        # test_log.add({'id': test_id, 'fake': label, 'prediction': prediction})

        explain(model, data, test_id)

    # test_log.to_csv(osp.join('explanations', 'test_log.csv'))


if __name__ == '__main__':
    main()

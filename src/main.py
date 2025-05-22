import argparse
import gc
import os
import os.path as osp
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from SPARQLWrapper import SPARQLWrapper, JSON
from torch.utils.data import random_split
from torch_geometric.data import DataListLoader
# from franz.openrdf.vocabulary import RDFS
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
from tqdm import tqdm

# from dataset.fnn_dataset import FNNWikidata5MDataset
from dataset.liar_raw_dataset import LiarRawDataset
from eval_helper import eval_deep
from model import Model
from only_text_model import OnlyTextModel
from src.dataset.fnn_dataset import FNNWikidata5MDataset
from util import clean_gpu

DATA_DIR = 'data'
DATASET = 'politifact'
MODEL_DIR = 'model'


#
# def example():
#     print("Available catalogs:")
#     for cat_name in server.listCatalogs():
#         if cat_name is None:
#             print('  - <root catalog>')
#         else:
#             print('  - ' + str(cat_name))
#
#     print(server.listCatalogs())
#
#     catalog = server.openCatalog('')
#     print("Available repositories in catalog '%s':" % catalog.getName())
#     for repo_name in catalog.listRepositories():
#         print('  - ' + repo_name)
#
#     # mode = Repository.OPEN
#     # my_repository = catalog.getRepository('actors', mode)
#     # my_repository.initialize()
#     #
#     # conn = my_repository.getConnection()
#     # print('Repository %s is up!' % my_repository.getDatabaseName())
#     # print('It contains %d statement(s).' % conn.size())
#
#     # indices = conn.listIndices()
#     # print('Current triple indices:', ', '.join(indices))
#
#     # with catalog.getRepository('actors', Repository.OPEN) as repo:
#     #     # Note: an explicit call to initialize() is not required
#     #     # when using the `with` statement.
#     #     with repo.getConnection() as conn:
#     #         print('Statements:', conn.size())
#
#     with ag_connect('actors', create=False, clear=False) as conn:
#         print('Statements:', conn.size())
#
#         query_string = "SELECT ?s ?p ?o  WHERE {?s ?p ?o .} LIMIT 10"
#
#         tuple_query = conn.prepareTupleQuery(QueryLanguage.SPARQL, query_string)
#         result = tuple_query.evaluate()
#
#         with result:
#             for binding_set in result:
#                 s = binding_set.getValue("s")
#                 p = binding_set.getValue("p")
#                 o = binding_set.getValue("o")
#                 print("%s %s %s" % (s, p, o))
#
#
# def loadData():
#     WIKIDATA5M_DIR = 'wikidata5m'
#     labels = dict()
#     labelSaved = set()
#     triplets = []
#
#     def read_labels(filename):
#         with open(os.path.join(WIKIDATA5M_DIR, filename), 'r', encoding="utf8") as f:
#             for line in f:
#                 terms = line.strip().split('\t')
#                 entity_id = terms[0]
#                 terms.pop(0)
#                 labels[entity_id] = terms
#
#     read_labels('wikidata5m_entity.txt')
#     read_labels('wikidata5m_relation.txt')
#
#     with open(os.path.join(WIKIDATA5M_DIR, 'wikidata5m_all_triplet.txt'), 'r') as f:
#         triplets = f.readlines()
#
#     print(len(labels), len(triplets))
#
#     wd = 'http://www.wikidata.org/entity/'
#     wdt = 'http://www.wikidata.org/prop/direct/'
#
#     LIMIT = 5000000 - 166497
#
#     with ag_connect('wikidata5m', create=True, clear=True) as conn:
#         countTriplets = 0
#         with tqdm(desc="Uploading triplets", total=LIMIT) as pbar:
#             for line in triplets:
#                 if countTriplets > LIMIT:
#                     exit(0)
#
#                 def save_label(s, s_id):
#                     global countTriplets
#                     if countTriplets + 1 >= LIMIT:
#                         return
#
#                     if s_id not in labelSaved:
#                         labelSaved.add(s_id)
#                         label = conn.createLiteral(labels[s_id][0])
#                         conn.add(s, RDFS.LABEL, label)
#                         countTriplets = countTriplets + 1
#                         pbar.update()
#
#                 subjId, predId, objId = line.strip().split('\t')
#
#                 subj = conn.createURI(wd + subjId)
#                 pred = conn.createURI(wdt + predId)
#                 obj = conn.createURI(wd + objId)
#
#                 save_label(subj, subjId)
#                 save_label(pred, predId)
#                 save_label(obj, objId)
#
#                 conn.add(subj, pred, obj)
#                 pbar.update()
#                 countTriplets += 1


# def uploadWikidata():
#     WIKIDATA5M_DIR = 'wikidata5m'
#
#     wd = Namespace('http://www.wikidata.org/entity/')
#     wdt = Namespace('http://www.wikidata.org/prop/direct/')
#
#     buffer = []
#     buffer_curr_size = 0
#     BUFFER_SIZE = 1600000
#     LABELS_LIMIT = 30
#
#     # print(RDFS.label)
#     # exit(0)
#
#     def split(list_a, chunk_size):
#         for i in range(0, len(list_a), chunk_size):
#             yield list_a[i:i + chunk_size]
#
#     def add_wd_triples(s, p, o):
#         save_line(f"wd:{s} wdt:{p} wd:{o}.")
#
#     def add_wd_labels(ns, e, labels):
#         escaped_labels = list(
#             map(lambda label: f"'{label.replace("\\", r"\\").replace('"', r'\"').replace("'", r"\'")}'", labels))
#         label_buffer = []
#         label_buffer_size = 0
#         for label in escaped_labels:
#             if label_buffer_size + len(label) > BUFFER_SIZE - 100 - len(ns) - len(e):
#                 save_line(f"""
#                             {ns}:{e} rdfs:label {', '.join(label_buffer)} .
#                             """)
#                 label_buffer = []
#                 label_buffer_size = 0
#             else:
#                 label_buffer_size = label_buffer_size + len(label) + 1
#                 label_buffer.append(label)
#
#         if label_buffer_size > 0:
#             save_line(f"""
#                         {ns}:{e} rdfs:label {', '.join(label_buffer)} .
#                         """)
#
#     def flush_buffer():
#         global buffer
#         global buffer_curr_size
#         if buffer_curr_size > 0:
#             save_lines(buffer)
#             buffer = []
#             buffer_curr_size = 0
#
#     def save_line(line):
#         global buffer
#         global buffer_curr_size
#         if buffer_curr_size + len(line) >= BUFFER_SIZE:
#             flush_buffer()
#         buffer_curr_size = buffer_curr_size + len(line)
#         buffer.append(line)
#
#     def save_lines(lines: list):
#         query = """
#         PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#         PREFIX wd: <http://www.wikidata.org/entity/>
#         PREFIX wdt: <http://www.wikidata.org/prop/direct/>
#         INSERT DATA
#               {""" + "\n".join(lines) + '}'
#         # print(len(query))
#         # with open("query.txt", "w", encoding="utf8") as f:
#         #     f.write(query)
#
#         sparql = SPARQLWrapper("http://localhost:7200/repositories/wikidata5m/statements")
#         sparql.method = POST
#         sparql.setReturnFormat(JSON)
#         sparql.setHTTPAuth(BASIC)
#         sparql.setCredentials('app', 'app')
#         sparql.queryType = "INSERT"
#         sparql.setQuery(query)
#         sparql.query()
#
#     def read_labels(filename, ns):
#         print("Reading {}".format(filename))
#         with open(os.path.join(WIKIDATA5M_DIR, filename), 'r', encoding="utf8") as f:
#             with tqdm(desc="Loading " + filename, total=os.stat(os.path.join(WIKIDATA5M_DIR, filename)).st_size) as bar:
#                 for line in f.readlines():
#                     terms = line.strip().split('\t')
#                     entity = terms[0]
#                     labels = terms[1:]
#                     add_wd_labels(ns, entity, labels)
#                     bar.update(len(line))
#
#                     # for label in terms[1:]:
#                     # g.add((entity, RDFS.label, Literal(label)))
#                     # add_triples()
#         flush_buffer()
#
#     def read_triplets(filename):
#         print("Reading {}".format(filename))
#         with open(os.path.join(WIKIDATA5M_DIR, filename), 'r', encoding="utf8") as f:
#             for line in tqdm(f.readlines()[:BUFFER_SIZE], desc="Loading " + filename):
#                 s, p, o = line.strip().split('\t')
#                 # g.add((wd[s_id], wdt[p_id], wd[o_id]))
#                 # add_triples(wd[s_id], wdt[p_id], wd[o_id])
#                 add_wd_triples(s, p, o)
#         flush_buffer()
#
#     def save_graph(graph, filename):
#         with open(os.path.join(WIKIDATA5M_DIR, filename), 'w', encoding="utf8") as fo:
#             fo.write(graph.serialize())
#
#     # with open(os.path.join(WIKIDATA5M_DIR, 'wikidata5m_all_triplet.txt'), 'r') as f:
#     #     triplets = f.readlines()
#
#     # print(len(labels), len(triplets))
#
#     # g = Graph()
#     # g.bind('wd', wd)
#     # g.bind('wdt', wdt)
#
#     # read_triplets('wikidata5m_all_triplet.txt')
#     # save_graph(g, 'wikidata5m-triplets.ttl')
#     read_labels('wikidata5m_entity.txt', 'wd')
#     # save_graph(g, 'wikidata5m-entities.ttl')
#     # read_labels('wikidata5m_relation.txt', wdt)
#     # save_graph(g, 'wikidata5m.ttl')


# setup_env_var('AGRAPH_HOST', 'https://ag1qzxuwy43npnmy.allegrograph.cloud/', 'Hostname')
# setup_env_var('AGRAPH_PORT', '443', 'Port')
# setup_env_var('AGRAPH_USER', 'admin', 'Username')
# setup_env_var('AGRAPH_PASSWORD', 'n8KKbtEOFkHOiGnRx8msou', 'Password')
#
# AGRAPH_HOST = os.environ.get('AGRAPH_HOST', 'localhost')
# AGRAPH_PORT = int(os.environ.get('AGRAPH_PORT', '10079'))
# AGRAPH_USER = os.environ.get('AGRAPH_USER', 'test')
# AGRAPH_PASSWORD = os.environ.get('AGRAPH_PASSWORD', 'xyzzy')
#
# print("Connecting to AllegroGraph server --",
#       "host:'%s' port:%s" % (AGRAPH_HOST, AGRAPH_PORT))
# server = AllegroGraphServer(AGRAPH_HOST, AGRAPH_PORT,
#                             AGRAPH_USER, AGRAPH_PASSWORD)

#
#     sparql.setQuery("""
#         select * where {
#     ?s ?p ?o .
# } limit 100
#     """)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
#
#     for result in results["results"]["bindings"]:
#         print(result["label"]["value"])
#
#     print('---------------------------')
#
#     for result in results["results"]["bindings"]:
#         print('%s: %s' % (result["label"]["xml:lang"], result["label"]["value"]))


def get_triples_from_annotation(ann):
    query = """
        SELECT ?item ?p ?o
        WHERE
        {
          VALUES ?url { <https://en.wikipedia.org/wiki/Donald_Trump>}
          ?url schema:about ?item. # Must be a cat
          ?item ?p ?o.
          FILTER(isUri(?o) && STRSTARTS(STR(?o), STR(wd:)) && isUri(?p) && STRSTARTS(STR(?p), STR(wdt:)))
        }
        """

    sparql = SPARQLWrapper(
        "https://query.wikidata.org/bigdata/namespace/wdq/sparql"
    )
    sparql.setReturnFormat(JSON)

    sparql.setQuery(query)

    try:
        ret = sparql.queryAndConvert()

        return [(row['item']['value'], row['p']['value'], row['o']['value']) for row in ret["results"]["bindings"]]
    except Exception as e:
        print(e)


@torch.no_grad()
def compute_test(loader, model, args, verbose=False):
    model.eval()
    loss_test = 0.0
    out_log = []
    for data in loader:
        if not args.multi_gpu:
            data = data.to(args.device)
        out = model(data)
        if args.multi_gpu:
            y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
        else:
            y = data.y
        # y = F.one_hot(y, num_classes=args.num_classes)


        # loss_test += F.nll_loss(out, y).item()
        loss_test += F.binary_cross_entropy_with_logits(out, y.float()).item()
        # loss_test += F.binary_cross_entropy(out, y.float()).item()

        if verbose:
        #     print(F.softmax(out, dim=1).cpu().numpy(), y.data.cpu().numpy())
        #     print(F.sigmoid(out.sigmoid()).cpu().numpy(), y.data.cpu().numpy())
        #     print(out.cpu().numpy(), y.data.cpu().numpy())
            print(out.sigmoid().round().int().data.cpu().numpy(), y.data.cpu().numpy())
        # out_log.append([F.softmax(out, dim=1).data.cpu().numpy(), y.data.cpu().numpy()])
        # out_log.append([out.data.cpu().numpy(), y.data.cpu().numpy()])
        # out_log.append([(F.sigmoid(out) > 0.5).int().data.cpu().numpy(), y.data.cpu().numpy()])
        # out_log.append([(out > 0.5).int().data.cpu().numpy(), y.data.cpu().numpy()])
        out_log.append([out.sigmoid().round().int().data.cpu().numpy(), y.data.cpu().numpy()])
        del data, out, y
    return eval_deep(out_log, loader), loss_test


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
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

dataset = FNNWikidata5MDataset(root='data', dataset=args.dataset, device=args.device, news_content=True, skip_processing=False)
# dataset = LiarRawDataset(root='data', device=args.device, skip_processing=False)

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

print(f"num_training: {num_training}, num_val:{num_val}, num_test:{num_test}, total:{len(dataset)}\n")

if args.multi_gpu:
    loader = DataListLoader
else:
    loader = DataLoader

# modelClass = OnlyTextModel()


def print_gpu_objects():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
                # for ref in gc.get_referrers(obj):
                #     # Should probably filter here to reduce the amount of data printed and avoid printing locals() as well
                #     print(ref)
        except:
            pass


def train():
    train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)

    # Model training
    model = createModel()
    if args.multi_gpu:
        model = DataParallel(model)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path('plots').mkdir(parents=True, exist_ok=True)
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pt"):
            os.remove(os.path.join(MODEL_DIR, f))

    max_f1 = 0
    val_loss_values = []
    best_epoch = 0
    best_model = None

    t = time.time()
    model.train()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    train_f1 = []
    val_f1 = []
    for epoch in tqdm(range(args.epochs)):
        loss_train = 0.0
        out_log = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            # print(torch.cuda.memory_summary())
            if not args.multi_gpu:
                data = data.to(args.device)

            # print(torch.cuda.memory_summary())
            out = model(data)
            if args.multi_gpu:
                y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
            else:
                y = data.y
            # y = F.one_hot(y, num_classes=args.num_classes).float()
            # loss = F.nll_loss(out, y)
            loss = F.binary_cross_entropy_with_logits(out, y.float())
            # loss = F.binary_cross_entropy(out, y.float())
            loss.backward()
            optimizer.step()
            # loss = loss.detach()
            loss_train += loss.item()
            # out_log.append([F.softmax(out, dim=1).data.cpu().numpy(), y.data.cpu().numpy()])
            # out_log.append([(F.sigmoid(out) > 0.5).int().data.cpu().numpy(), y.data.cpu().numpy()])
            # out_log.append([(out > 0.5).int().data.cpu().numpy(), y.data.cpu().numpy()])
            out_log.append([out.sigmoid().round().int().data.cpu().numpy(), y.data.cpu().numpy()])

            data.detach()
            data.grad = None
            # data.storage().resize_(0)
            del out, y, loss, data
            clean_gpu(args.device)
            # print(torch.cuda.memory_summary())
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size())
            #     except:
            #         pass
        acc_train, f1_train, precision_train, recall_train = eval_deep(out_log, train_loader)
        [acc_val, f1_val, precision__val, recall_val], loss_val = compute_test(val_loader, model, args)
        train_loss.append(loss_train)
        val_loss.append(loss_val)
        train_acc.append(acc_train)
        val_acc.append(acc_val)
        train_f1.append(f1_train)
        val_f1.append(f1_val)
        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f}, f1_train: {f1_train:.4f}, '
              f' recall_train: {recall_train:.4f},'
              f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f}, f1_val: {f1_val:.4f},'
              f' recall_val: {recall_val:.4f}')

        if f1_val >= max_f1:
            max_f1 = f1_val
            best_epoch = epoch
            best_model = deepcopy(model.state_dict())

        torch.save(model.state_dict(), f'model/model-{epoch}.pt')

    print(f"Best epoch: {best_epoch}")
    torch.save(best_model, f'model/model.pt')

    epochs_range = range(args.epochs)
    plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig(osp.join('plots', 'accuracy.png'))

    plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_f1, label='Training f1')
    plt.plot(epochs_range, val_f1, label='Validation f1')
    plt.legend(loc='lower right')
    plt.xticks(epochs_range, rotation=90)
    plt.title('Training and Validation f1')
    plt.savefig(osp.join('plots', 'f1.png'))

    plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xticks(epochs_range, rotation=90)
    plt.title('Training and Validation Loss')
    plt.savefig(osp.join('plots', 'loss.png'))


def test(epoch=None):
    test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)
    model = createModel()
    if args.multi_gpu:
        model = DataParallel(model)

    if epoch:
        modelPath = f'model/model-{epoch}.pt'
    else:
        modelPath = f'model/model.pt'
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    model = model.to(args.device)

    [acc, f1, precision, recall], test_loss = compute_test(test_loader, model, args, verbose=False)
    print(f'Test set results: acc: {acc:.4f}, f1: {f1:.4f}, '
          f'precision: {precision:.4f}, recall: {recall:.4f}')


def createModel():
    model = Model(args, concat=args.concat)
    return model
    # return OnlyTextModel(args)


if __name__ == '__main__':
    # dataset = FNNWikidata5MDataset(root=DATA_DIR, dataset=DATASET)
    # train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    #
    train()
    clean_gpu(args.device)
    test()

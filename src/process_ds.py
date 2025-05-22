import json
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

raw_dir = os.path.join('data', 'raw')
processed_dir = os.path.join('data', 'processed')
res_dir = os.path.join(processed_dir, 'res')
log_path = os.path.join(processed_dir, f"politifact_log.csv")
content_dir = os.path.join(raw_dir, 'politifact')


device = 'cuda:0'


def invert_label(data):
    data.y = 1 - data.y
    return data


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoder = BertModel.from_pretrained('bert-base-uncased').to(device)
encoder.eval()


def encode_short(texts, tokenizer, encoder, device='cuda:0'):
    tokens_batch = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=10).to(
        device)  # padding='max_length', truncation=True, max_length=5
    outputs_list = encoder(**tokens_batch, output_hidden_states=True)
    embeddings = outputs_list.last_hidden_state[:, 0].detach().cpu()


def encode_long(text, tokenizer, encoder, device='cuda:0'):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        device)  # padding='max_length', truncation=True, max_length=5
    outputs = encoder(**tokens, output_hidden_states=True)
    embedding = outputs.last_hidden_state[:, 0].detach().cpu()
    return embedding


def add_text_embedding(data, test_id):

    if data.y == 0:
        label = 'real'
    else:
        label = 'fake'

    content_path = os.path.join(content_dir, label, test_id, 'news content.json')

    has_content = False
    node_feature = None
    if os.path.isfile(content_path):
        with open(content_path) as f:
            news_content = json.load(f)
            if news_content['text']:
                has_content = True
                content_text = news_content['text']
                embedding = encode_long(content_text, tokenizer, encoder)
                # data['content'] = embedding[0]
                node_feature =  embedding[0]

    if node_feature is None:
        # data['content'] = torch.zeros(768)
        node_feature = torch.zeros(768)

    data.x[0] = torch.cat([torch.ones(1), node_feature])

    return data

def change_dtype(data):
    data.x = data.x.to(torch.float32)
    return data

def add_content(data):
    data.content = data.x[0]
    return data


if __name__ == '__main__':

    Path(res_dir).mkdir(parents=True, exist_ok=True)
    for file in tqdm(os.listdir(processed_dir)):
        if file.startswith('data_') and file.endswith('.pt'):
            data = torch.load(os.path.join(processed_dir, file))
            # data = invert_label(data)
            data = add_text_embedding(data, file[5:-3])
            # data = change_dtype(data)
            torch.save(data, os.path.join(res_dir, file))

# -*- coding: utf-8 -*-


import argparse
import gzip
import json
import pickle as pkl
import random
import re
import sys
import time

import nltk
import numpy as np
import scipy.sparse as sp
from sklearn.utils import shuffle

random.seed(44)
np.random.seed(44)

'''
Config:
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cola')
parser.add_argument('--kg', type=str, default='WN18')
parser.add_argument('--sw', type=int, default=0)

args = parser.parse_args()
dataset = args.dataset
kg = args.kg
del_stop_words = True if args.sw == 1 else False

task = 'MLM'

dataset_list = {'sst-2', 'cola', 'fbqa', 'rte', 'wiki', 'amazon'}
folder_names = {'cola': 'CoLA', 'sst-2': 'SST-2', 'fbqa': 'FreebaseQA', 'rte': 'RTE', 'amazon': 'Amazon',
                'wiki': 'Wiki2'}
kgs = {'WN11', 'WN18'}

if dataset not in dataset_list:
    sys.exit("Dataset choice error!")
if kg not in kgs:
    sys.exit("Knowledge Graph choice error!")

will_dump_objects = True
dump_dir = 'data/dump/' + task
if del_stop_words:
    freq_min_for_word_choice = 5
else:
    freq_min_for_word_choice = 1

valid_data_taux = 0.05
test_data_taux = 0.10

# word co-occurence with context windows
window_size = 20

# tfidf_mode = 'only_tf'
tfidf_mode = 'all_tfidf'

use_bert_tokenizer_at_clean = True

bert_model_scale = 'bert-base-uncased'
bert_lower_case = True

print('---data prepare configure---')
print('Data set: ', dataset, 'freq_min_for_word_choice', freq_min_for_word_choice, 'window_size', window_size)
print('del_stop_words', del_stop_words, 'use_bert_tokenizer_at_clean', use_bert_tokenizer_at_clean)
print('bert_model_scale', bert_model_scale, 'bert_lower_case', bert_lower_case)
print('\n')

start = time.time()
import pandas as pd


def del_http_user_tokenize(tweet):
    # delete [ \t\n\r\f\v]
    space_pattern = r'\s+'
    url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                 r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = r'@[\w\-]+'
    tweet = re.sub(space_pattern, ' ', tweet)
    tweet = re.sub(url_regex, '', tweet)
    tweet = re.sub(mention_regex, '', tweet)
    return tweet


dataset_glue_dir = 'data/GLUE/'
dataset_general_dir = 'data/'

if dataset == 'sst-2':
    label2idx = {'0': 0, '1': 1}
    idx2label = {0: '0', 1: '1'}
    train_df = pd.read_csv(dataset_glue_dir + 'SST-2/train.tsv', encoding='utf-8', header=0, sep='\t')
    train_df = shuffle(train_df)
    valid_df = pd.read_csv(dataset_glue_dir + 'SST-2/dev.tsv', encoding='utf-8', header=0, sep='\t')
    valid_df = shuffle(valid_df)
    test_df = pd.read_csv(dataset_glue_dir + 'SST-2/test.tsv', encoding='utf-8', header=0, sep='\t')
    test_df = shuffle(test_df)
    train_size = train_df['sentence'].count()
    valid_size = valid_df['sentence'].count()
    test_size = len(test_df['index'])
    print('SST-2 train Total:', train_size, 'valid Total:', valid_size, 'test Total:', test_size)
    corpus = pd.concat((train_df['sentence'], valid_df['sentence'], test_df['sentence']))
    y = pd.concat((train_df['label'], valid_df['label'])).values
    test_indices = test_df['index'].values
    y_prob = np.eye(len(y), len(label2idx))[y]
    corpus_size = len(corpus)

    trains, valids, tests = train_df['sentence'], valid_df['sentence'], test_df['sentence']
    trains_y, valids_y = train_df['label'], valid_df['label']

elif dataset == 'cola':
    label2idx = {'0': 0, '1': 1}
    idx2label = {0: '0', 1: '1'}
    train_df = pd.read_csv(dataset_glue_dir + 'CoLA/train.tsv', encoding='utf-8', header=None, sep='\t')
    train_df = shuffle(train_df)
    valid_df = pd.read_csv(dataset_glue_dir + 'CoLA/dev.tsv', encoding='utf-8', header=None, sep='\t')
    valid_df = shuffle(valid_df)
    test_df = pd.read_csv(dataset_glue_dir + 'CoLA/test.tsv', encoding='utf-8', header=0, sep='\t')
    test_df = shuffle(test_df)
    train_size = train_df[1].count()
    valid_size = valid_df[1].count()
    test_size = len(test_df['index'])
    print('CoLA train Total:', train_size, 'valid Total:', valid_size, 'test Total:', test_size)
    corpus = pd.concat((train_df[3], valid_df[3], test_df['sentence']))
    y = pd.concat((train_df[1], valid_df[1])).values
    test_indices = test_df['index'].values
    y_prob = np.eye(len(y), len(label2idx))[y]
    corpus_size = len(corpus)

    trains, valids, tests = train_df[3], valid_df[3], test_df['sentence']
    trains_y, valids_y = train_df[1], valid_df[1]

elif dataset == 'amazon':
    cats = ['All_Beauty_5',
            'Industrial_and_Scientific_5']  # , 'Musical_Instruments_5', 'Software_5', 'Arts_Crafts_and_Sewing_5']#, 'Gift_Cards_5', 'Sports_and_Outdoors_5', 'Cell_Phones_and_Accessories_5']

    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield json.loads(l)

    def getDF():
        i = 0
        df = {}
        for f in cats:
            path = '{}Amazon/{}.json.gz'.format(dataset_general_dir, f)
            for d in parse(path):
                df[i] = d
                i += 1
            print(f)
        return pd.DataFrame.from_dict(df, orient='index')

    df = getDF().dropna()

    label2idx = {'0': 0, '1': 1}
    idx2label = {0: '0', 1: '1'}
    df = shuffle(df)
    corpus = df['reviewText']
    corpus_size = len(corpus)
    train_size = int(0.6 * corpus_size)
    valid_size = int(0.2 * corpus_size)
    test_size = corpus_size - (train_size + valid_size)
    trains = corpus.iloc[:train_size]
    valids = corpus.iloc[train_size:train_size + valid_size]
    tests = corpus.iloc[train_size + valid_size:]
    y = np.concatenate((np.zeros(train_size), np.ones(valid_size)))
    np.random.shuffle(y)
    y = y.astype(int)
    trains_y, valids_y = y[:train_size], y[train_size:]
    y_prob = np.eye(len(y), len(label2idx))[y]
    test_indices = np.arange(test_size)

elif dataset == 'wiki':
    def getDF(method):
        i = 0
        df = {}
        path = '{}wikitext-2/wiki.{}.tokens'.format(dataset_general_dir, method)
        with open(path, 'r+') as f:
            for l in f:
                line = l.strip()
                if line == '' or (line.startswith('=') and line.endswith('=')):
                    continue
                df[i] = line
                i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    valid_df = getDF('valid')
    train_df = getDF('train')
    test_df = getDF('test')
    train_df = shuffle(train_df)
    valid_df = shuffle(valid_df)
    test_df = shuffle(test_df)

    label2idx = {'0': 0, '1': 1}
    idx2label = {0: '0', 1: '1'}

    train_size = len(train_df[0])
    valid_size = len(valid_df[0])
    test_size = len(test_df[0])
    print('Wikitext-2 train Total:', train_size, 'valid Total:', valid_size, 'test Total:', test_size)
    corpus = pd.concat((train_df[0], valid_df[0], test_df[0]))
    y = np.concatenate((np.zeros(train_size), np.ones(valid_size)))
    np.random.shuffle(y)
    y = y.astype(int)
    trains_y, valids_y = y[:train_size], y[train_size:]
    y_prob = np.eye(len(y), len(label2idx))[y]
    test_indices = np.arange(test_size)
    corpus_size = len(corpus)

    trains, valids, tests = train_df[0], valid_df[0], test_df[0]
    trains_y, valids_y = y[:train_size], y[train_size:]

doc_content_list = []
for t in corpus:
    doc_content_list.append(del_http_user_tokenize(t))
max_len_seq = 0
max_len_seq_idx = -1
min_len_seq = 1000
min_len_seq_idx = -1
sen_len_list = []
for i, seq in enumerate(doc_content_list):
    seq = seq.split()
    sen_len_list.append(len(seq))
    if len(seq) < min_len_seq:
        min_len_seq = len(seq)
        min_len_seq_idx = i
    if len(seq) > max_len_seq:
        max_len_seq = len(seq)
        max_len_seq_idx = i
print('Statistics for original text: max_len%d,id%d, min_len%d,id%d, avg_len%.2f' \
      % (max_len_seq, max_len_seq_idx, min_len_seq, min_len_seq_idx, np.array(sen_len_list).mean()))


'''
Remove stop words from tweets
'''
print('Remove stop words from tweets...')

if del_stop_words:
    from nltk.corpus import stopwords

    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
else:
    stop_words = {}
print('Stop_words:', stop_words)


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = " ".join(re.split("[^a-zA-Z]", string.lower())).strip()
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


tmp_word_freq = {}  # to remove rare words
new_doc_content_list = []

# use bert_tokenizer in order to split the sentence
if use_bert_tokenizer_at_clean:
    print('Use bert_tokenizer for seperate words to bert vocab')
    from pytorch_pretrained_bert import BertTokenizer

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=bert_lower_case)

for doc_content in doc_content_list:
    new_doc = clean_str(doc_content)
    if use_bert_tokenizer_at_clean:
        sub_words = bert_tokenizer.tokenize(new_doc)
        sub_doc = ' '.join(sub_words).strip()
        new_doc = sub_doc
    new_doc_content_list.append(new_doc)
    for word in new_doc.split():
        if word in tmp_word_freq:
            tmp_word_freq[word] += 1
        else:
            tmp_word_freq[word] = 1

doc_content_list = new_doc_content_list

clean_docs = []
count_void_doc = 0
for i, doc_content in enumerate(doc_content_list):
    words = doc_content.split()
    doc_words = []
    for word in words:
        if dataset in dataset_list:
            doc_words.append(word)
        elif word not in stop_words and tmp_word_freq[word] >= freq_min_for_word_choice:
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    if doc_str == '':
        count_void_doc += 1
        print('No.', i, 'is a empty doc after treat, replaced by \'%s\'. original:%s' % (doc_str, doc_content))
    clean_docs.append(doc_str)

print('Total', count_void_doc, ' docs are empty.')

min_len = 10000
min_len_id = -1
max_len = 0
max_len_id = -1
aver_len = 0

for i, line in enumerate(clean_docs):
    temp = line.strip().split()
    aver_len = aver_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
        min_len_id = i
    if len(temp) > max_len:
        max_len = len(temp)
        max_len_id = i

aver_len = 1.0 * aver_len / len(clean_docs)
print('After tokenizer:')
print('Min_len : ' + str(min_len) + ' id: ' + str(min_len_id))
print('Max_len : ' + str(max_len) + ' id: ' + str(max_len_id))
print('Average_len : ' + str(aver_len))

'''
Build multi-graph
'''
print('Build multi-graph...')

if dataset in dataset_list:
    shuffled_clean_docs = clean_docs
    train_docs = shuffled_clean_docs[:train_size]
    valid_docs = shuffled_clean_docs[train_size:train_size + valid_size]
    train_valid_docs = shuffled_clean_docs[:train_size + valid_size]
    train_y = y[:train_size]
    valid_y = y[train_size:train_size + valid_size]
    train_y_prob = y_prob[:train_size]
    valid_y_prob = y_prob[train_size:train_size + valid_size]

# build vocab using the whole corpus + specific BERT vocabs
word_set = set(['[CLS]', '[MASK]', '[SEP]'])
for doc_words in shuffled_clean_docs:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
vocab = list(word_set)
vocab_size = len(vocab)

all_vocab_ids_list = bert_tokenizer.convert_tokens_to_ids(vocab)

vocab_map = {}
all_vocab_ids_map = {}
for i in range(vocab_size):
    vocab_map[vocab[i]] = i
    all_vocab_ids_map[vocab[i]] = all_vocab_ids_list[i]

# build vocab_train_valid
word_set_train_valid = set()
for doc_words in train_valid_docs:
    words = doc_words.split()
    for word in words:
        word_set_train_valid.add(word)
vocab_train_valid = list(word_set_train_valid)
vocab_train_valid_size = len(vocab_train_valid)

for_idf_docs = shuffled_clean_docs
word_doc_list = {}
for i in range(len(for_idf_docs)):
    doc_words = for_idf_docs[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

print('Calculate First isomerous adj and First isomorphic vocab adj, get word-word PMI values')

adj_y = np.hstack((train_y, np.zeros(vocab_size), valid_y))
adj_y_prob = np.vstack(
    (train_y_prob, np.zeros((vocab_size, len(label2idx)), dtype=np.float32), valid_y_prob))

windows = []
for doc_words in train_valid_docs:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

print('Train_valid size:', len(train_valid_docs), 'Window number:', len(windows))

word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    appeared = set()
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = vocab_map[word_i]
            word_j = window[j]
            word_j_id = vocab_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in appeared:
                continue
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            appeared.add(word_pair_str)
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in appeared:
                continue
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            appeared.add(word_pair_str)

from math import log

row = []
col = []
weight = []
tfidf_row = []
tfidf_col = []
tfidf_weight = []
vocab_adj_row = []
vocab_adj_col = []
vocab_adj_weight = []

num_window = len(windows)
tmp_max_npmi = 0
tmp_min_npmi = 0
tmp_max_pmi = 0
tmp_min_pmi = 0
for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))

    npmi = log(1.0 * word_freq_i * word_freq_j / (num_window * num_window)) / log(1.0 * count / num_window) - 1
    if npmi > tmp_max_npmi: tmp_max_npmi = npmi
    if npmi < tmp_min_npmi: tmp_min_npmi = npmi
    if pmi > tmp_max_pmi: tmp_max_pmi = pmi
    if pmi < tmp_min_pmi: tmp_min_pmi = pmi
    if pmi > 0:
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)
    if npmi > 0:
        vocab_adj_row.append(i)
        vocab_adj_col.append(j)
        vocab_adj_weight.append(npmi)
print('max_pmi:', tmp_max_pmi, 'min_pmi:', tmp_min_pmi)
print('max_npmi:', tmp_max_npmi, 'min_npmi:', tmp_min_npmi)

print('Calculate doc-word tf-idf weight')

n_docs = len(shuffled_clean_docs)
doc_word_freq = {}
for doc_id in range(n_docs):
    doc_words = shuffled_clean_docs[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = vocab_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(n_docs):
    doc_words = shuffled_clean_docs[i]
    words = doc_words.split()
    doc_word_set = set()
    tfidf_vec = []
    for word in words:
        if word in doc_word_set:
            continue
        j = vocab_map[word]
        key = str(i) + ',' + str(j)
        tf = doc_word_freq[key]
        tfidf_row.append(i)
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        tfidf_col.append(j)
        col.append(train_size + j)
        # smooth
        idf = log((1.0 + n_docs) / (1.0 + word_doc_freq[vocab[j]])) + 1.0

        if tfidf_mode == 'only_tf':
            tfidf_vec.append(tf)
        else:
            tfidf_vec.append(tf * idf)
        doc_word_set.add(word)
    if len(tfidf_vec) > 0:
        weight.extend(tfidf_vec)
        tfidf_weight.extend(tfidf_vec)

'''
Knowledge Graph addition
'''
kg_id2ent = {}
kg_ent2id = {}

kg_entities = np.loadtxt('data/{}/entity2id.txt'.format(kg), dtype=np.str, skiprows=1, delimiter='\t')
cnt = 0
for i in range(len(kg_entities)):
    ent = ' '.join(clean_str(kg_entities[i, 0]).split()[:-1]).strip()
    sub_ents = bert_tokenizer.tokenize(ent)
    kg_id2ent[int(kg_entities[i, 1])] = sub_ents

kg_links = np.loadtxt('data/{}/triple2id.txt'.format(kg), dtype=np.str, skiprows=1, delimiter='\t')
triple_i, triple_j = 0, 1

print('loaded KG')
print('Calculate KG matrix')
kg_row = []
kg_col = []
kg_weight = []
for idx, subs in kg_id2ent.items():
    for sub in subs:
        if sub in vocab_map:
            kg_row.append(vocab_map[sub])
            kg_col.append(idx)
            kg_weight.append(1)
for idx in range(len(kg_links)):
    i = np.long(kg_links[idx, triple_i])
    j = np.long(kg_links[idx, triple_j])
    for subi in kg_id2ent[i]:
        if subi in vocab_map:
            kg_row.append(vocab_map[subi])
            kg_col.append(j)
            kg_weight.append(1)
    for subj in kg_id2ent[j]:
        if subj in vocab_map:
            kg_row.append(vocab_map[subj])
            kg_col.append(i)
            kg_weight.append(1)

kg_adj = sp.csr_matrix((kg_weight, (kg_row, kg_col)), shape=(vocab_size, len(kg_id2ent)), dtype=np.float32).tolil()
for i in range(kg_adj.shape[0]):
    norm = np.linalg.norm(kg_adj.data[i])
    if norm > 0:
        kg_adj.data[i] /= norm

kg_adj = kg_adj.dot(kg_adj.T)

'''
Assemble graph matrices and dump to files
'''
node_size = vocab_size + corpus_size
print('node size', node_size)

vocab_adj_mi = sp.csr_matrix((vocab_adj_weight, (vocab_adj_row, vocab_adj_col)), shape=(vocab_size, vocab_size),
                             dtype=np.float32)
vocab_adj_mi.setdiag(1.0)

print('Calculate isomorphic vocab adjacency matrix using doc\'s tf-idf...')
tfidf_all = sp.csr_matrix((tfidf_weight, (tfidf_row, tfidf_col)), shape=(corpus_size, vocab_size), dtype=np.float32)
tfidf_train = tfidf_all[:train_size]
tfidf_valid = tfidf_all[train_size:train_size + valid_size]
tfidf_test = tfidf_all[train_size + valid_size:]
tfidf_X_list = [tfidf_train, tfidf_valid, tfidf_test]
vocab_tfidf = tfidf_all.T.tolil()
for i in range(vocab_size):
    norm = np.linalg.norm(vocab_tfidf.data[i])
    if norm > 0:
        vocab_tfidf.data[i] /= norm
vocab_adj_tf = vocab_tfidf.dot(vocab_tfidf.T)

folder = folder_names[dataset]

# dump objects
if will_dump_objects:
    print('Dump objects...')

    with open(dump_dir + "/%s/labels_%s" % (folder, kg), 'wb') as f:
        pkl.dump([label2idx, idx2label], f)

    with open(dump_dir + "/%s/vocab_map_%s" % (folder, kg), 'wb') as f:
        pkl.dump(vocab_map, f)
    with open(dump_dir + "/%s/vocab_bert_ids_map_%s" % (folder, kg), 'wb') as f:
        pkl.dump(all_vocab_ids_map, f)

    with open(dump_dir + "/%s/vocab_%s" % (folder, kg), 'wb') as f:
        pkl.dump(vocab, f)

    with open(dump_dir + "/%s/y_%s" % (folder, kg), 'wb') as f:
        pkl.dump(y, f)
    with open(dump_dir + "/%s/y_prob_%s" % (folder, kg), 'wb') as f:
        pkl.dump(y_prob, f)
    with open(dump_dir + "/%s/train_y_%s" % (folder, kg), 'wb') as f:
        pkl.dump(train_y, f)
    with open(dump_dir + "/%s/train_y_prob_%s" % (folder, kg), 'wb') as f:
        pkl.dump(train_y_prob, f)
    with open(dump_dir + "/%s/valid_y_%s" % (folder, kg), 'wb') as f:
        pkl.dump(valid_y, f)
    with open(dump_dir + "/%s/valid_y_prob_%s" % (folder, kg), 'wb') as f:
        pkl.dump(valid_y_prob, f)
    with open(dump_dir + "/%s/test_indices_%s" % (folder, kg), 'wb') as f:
        pkl.dump(test_indices, f)

    with open(dump_dir + "/%s/tfidf_list_%s" % (folder, kg), 'wb') as f:
        pkl.dump(tfidf_X_list, f)
    with open(dump_dir + "/%s/pmi_adj_%s" % (folder, kg), 'wb') as f:
        pkl.dump(vocab_adj_mi, f)
    with open(dump_dir + "/%s/tf_adj_%s" % (folder, kg), 'wb') as f:
        pkl.dump(vocab_adj_tf, f)
    with open(dump_dir + "/%s/kg_adj_%s" % (folder, kg), 'wb') as f:
        pkl.dump(kg_adj, f)

    with open(dump_dir + "/%s/shuffled_clean_docs_%s" % (folder, kg), 'wb') as f:
        pkl.dump(shuffled_clean_docs, f)

print('Data prepared, spend %.2f s' % (time.time() - start))

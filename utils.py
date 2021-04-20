# -*- coding: utf-8 -*-

import random

import numpy as np
import scipy.sparse as sp
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


# report Hits@ks based on predictions and true values
def hits_at_k(pred, truth, ks):
    maxk = max(ks)
    _, max_idx = torch.topk(pred, k=maxk)
    hits = {}
    for k in ks:
        correct = max_idx[:, :k].eq(truth.view(-1, 1).expand_as(max_idx[:, :k]))
        hits[k] = (correct.float().sum(1, keepdim=True) > 0).float().sum()
    return hits


# to print
def get_hits_str(hits):
    s = ''
    for k, hit in hits.items():
        s += 'Hits@%d : %.3f ' % (k, hit * 100)
    return s


def savefig(losses, args):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].plot(losses['train'])
    ax[0].set_title('train loss')
    ax[1].plot(losses['valid'])
    ax[1].set_title('valid loss')
    ax[2].plot(losses['test'])
    ax[2].set_title('test loss')

    model_name = args['model']
    dataset = args['ds']
    graph = args['graph_mode']
    run = args['run']
    max_epoch = args['max_epoch']
    dyn = args['dyn']

    file = 'results/loss_figs/{}_model_{}_graph[{}]_run{}_dyn{}_max_epoch{}.png'.format(model_name,
                                                                                        dataset, graph,
                                                                                        run, dyn, max_epoch)
    fig.savefig(file)


def print_and_write(s, args):
    model_name = args['model']
    dataset = args['ds']
    graph = args['graph_mode']
    run = args['run']
    max_epoch = args['max_epoch']
    dyn = args['dyn']
    file = 'results/continuous_report/{}_model_{}_graph[{}]_run{}_dyn{}_max_epoch{}.txt'.format(model_name, dataset,
                                                                                                graph, run, dyn,
                                                                                                max_epoch)
    with open(file, 'a+') as f:
        f.write(s + '\n')
    print(s)


def clean_tweet_tokenize(string):
    tknzr = TweetTokenizer(reduce_len=True, preserve_case=False, strip_handles=False)
    tokens = tknzr.tokenize(string.lower())
    return ' '.join(tokens).strip()


# to normalize the multi-graphs
def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def sparse_scipy2torch(coo_sparse):
    i = torch.LongTensor(np.vstack((coo_sparse.row, coo_sparse.col)))
    v = torch.from_numpy(coo_sparse.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo_sparse.shape))


class InputExample(object):
    def __init__(self, guid, text):
        self.guid = guid
        self.text = text


class InputFeatures(object):
    def __init__(self, guid, tokens, input_ids, vocab_ids, input_mask, segment_ids, true_input_ids,
                 just_masked_input_index):
        self.guid = guid
        self.tokens = tokens
        self.input_ids = input_ids
        self.vocab_ids = vocab_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.true_input_ids = true_input_ids
        self.just_masked_input_index = just_masked_input_index


# mask a random token in a sentence
def example2feature(example, tokenizer, vocab_map, max_seq_len, vocab_bert_ids_map):
    tokens = example.text.split()

    if len(tokens) > max_seq_len - 1:
        print('GUID: %d, Sentence is too long: %d' % (example.guid, len(tokens)))
        tokens = tokens[:(max_seq_len - 1)]

    true_tokens = ["[CLS]"] + tokens + ["[SEP]"]
    true_input_ids = tokenizer.convert_tokens_to_ids(true_tokens)
    masked_index = random.sample(range(len(tokens)), 1)
    for i, idx in enumerate(masked_index):
        tokens[idx] = "[MASK]"
        masked_index[i] += 1  # due to CLS addition.

    vocab_ids = []
    for w in tokens:
        vocab_ids.append(vocab_map[w])

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    masked_input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(masked_input_ids)

    feat = InputFeatures(
        guid=example.guid,
        tokens=tokens,
        input_ids=masked_input_ids,
        vocab_ids=vocab_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        true_input_ids=true_input_ids,
        just_masked_input_index=masked_index,
    )
    return feat


class CorpusDataset(Dataset):
    def __init__(self, examples, tokenizer, vocab_map, vocab_bert_ids_map, max_seq_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.vocab_map = vocab_map
        self.vocab_bert_ids_map = vocab_bert_ids_map
        id2bert = [0] * len(vocab_map)
        for k in vocab_bert_ids_map.keys():
            id2bert[vocab_map[k]] = vocab_bert_ids_map[k]
        self.id2bert = torch.tensor(id2bert, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(self.examples[idx], self.tokenizer, self.vocab_map, self.max_seq_len,
                               self.vocab_bert_ids_map)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.vocab_ids, feat.true_input_ids, feat.just_masked_input_index, feat.guid

    def pad(self, batch):
        vocab_size = len(self.vocab_map)
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f_collect = lambda x: [sample[x] for sample in batch]
        f_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
        f_pad2 = lambda x, seqlen: [[-1] + sample[x] + [-1] * (seqlen - len(sample[x]) - 1) for sample in batch]

        batch_input_ids = torch.tensor(f_pad(0, maxlen), dtype=torch.long)
        batch_input_mask = torch.tensor(f_pad(1, maxlen), dtype=torch.long)
        batch_segment_ids = torch.tensor(f_pad(2, maxlen), dtype=torch.long)

        batch_vocab_ids_paded = np.array(f_pad2(3, maxlen)).reshape(-1)
        batch_true_input_ids = torch.tensor(f_pad(4, maxlen), dtype=torch.long)
        batch_indices = torch.tensor(f_collect(6), dtype=torch.long)

        batch_just_masked_input_index = torch.tensor(f_collect(5), dtype=torch.long)
        batch_swop_eye = torch.eye(vocab_size + 1)[batch_vocab_ids_paded][:, :-1]
        batch_swop_eye = batch_swop_eye.view(len(batch), -1, vocab_size)

        return batch_input_ids, batch_input_mask, batch_segment_ids, batch_swop_eye, batch_true_input_ids, batch_just_masked_input_index

import argparse
import gc
import os
import pickle as pkl
import time

import torch

from model import MGBERT_MLM
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cola')
parser.add_argument('--kg', type=str, default='WN18')
parser.add_argument('--just-masked', type=int, default=1)  # loss on just masked tokens
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--graph-mode', type=str, default='0')  # 123 : tf-mi-kg
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--sw', type=int, default='0')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--dyn', type=float, default=0.0)  # dyn = 0: static, 0 < dyn < 1: dynamic
parser.add_argument('--lr', type=float, default=0)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--model', type=str, default='MG-BERT')
parser.add_argument('--data', type=str, default='valid')

parser.add_argument('--kg-scale', type=float, default=0.01)
parser.add_argument('--tf-scale', type=float, default=0.001)
parser.add_argument('--mi-scale', type=float, default=0.001)

parser.add_argument('--batch', type=int, default=64)

args = parser.parse_args()

dataset = args.dataset
kg = args.kg
run = args.run
model_name = args.model
batch_size = args.batch
stop_words = True if args.sw == 1 else False

lambda_dyn = args.dyn

bert_model_scale = 'bert-base-uncased'
do_lower_case = True

dataset_list = {'sst-2', 'cola', 'rte', 'wiki', 'brown'}
folder_names = {'cola': 'CoLA', 'sst-2': 'SST-2', 'rte': 'RTE', 'wiki': 'Wiki2', 'brown': 'Brown'}

adj_mi_threshold = 0.2
adj_tf_threshold = 0
adj_kg_threshold = 0

data_dir = 'data/dump'
output_dir = 'output/'
task = 'MLM'

cuda_available = torch.cuda.is_available()
# cuda_available = False
if cuda_available:
    torch.cuda.manual_seed_all(int(time.time()))
device = torch.device("cuda:0" if cuda_available else "cpu")

assert dataset in dataset_list
if dataset == 'sst-2':
    learning_rate0 = 1e-5 if args.lr <= 0 else args.lr  # 2e-5
    report_each_epoch = 10
    sent = 200
elif dataset == 'cola':
    learning_rate0 = 8e-6 if args.lr <= 0 else args.lr  # 2e-5
    report_each_epoch = 50
    sent = 200
elif dataset == 'brown':
    learning_rate0 = 8e-6 if args.lr <= 0 else args.lr  # 2e-5
    report_each_epoch = 20
    sent = 250

max_len = sent

tmp_folder = folder_names[dataset]
objects = []
names = ['test_indices', 'labels', 'train_y', 'train_y_prob', 'valid_y', 'valid_y_prob', 'shuffled_clean_docs']
for i in range(len(names)):
    datafile = "%s/%s/%s/%s_%s" % (data_dir, task, folder_names[dataset], names[i], kg)
    with open(datafile, 'rb') as f:
        objects.append(pkl.load(f, encoding='latin1'))
test_indices, lables_list, train_y, train_y_prob, valid_y, valid_y_prob, shuffled_clean_docs = tuple(objects)

tmp_folder = folder_names[dataset]
objects = []
names = ['tf_adj', 'pmi_adj', 'kg_adj', 'vocab_map', 'vocab_bert_ids_map']
for i in range(len(names)):
    datafile = "%s/%s/%s/%s_%s" % (data_dir, task, tmp_folder, names[i], kg)
    with open(datafile, 'rb') as f:
        objects.append(pkl.load(f, encoding='latin1'))
adj_tf, adj_mi, adj_kg, vocab_map, vocab_bert_ids_map = tuple(objects)

label2idx = lables_list[0]
idx2label = lables_list[1]
test_indices = np.array(test_indices)

test_y = np.zeros_like(test_indices)  # just an array
y = np.hstack((train_y, valid_y, test_y))
vocab_size = len(vocab_map)
train_size = len(train_y)
valid_size = len(valid_y)
test_size = len(test_y)
# for when we are working on a dataset with fixed indices
indices = np.vstack((np.array(list(range(train_size + valid_size))).reshape(-1, 1), test_indices.reshape(-1, 1)))

examples = []
for i, ts in enumerate(shuffled_clean_docs):
    ex = InputExample(indices[i], ts.strip())
    examples.append(ex)

indexs = np.arange(0, len(examples))
train_examples = [examples[i] for i in indexs[:train_size]]
valid_examples = [examples[i] for i in indexs[train_size:train_size + valid_size]]
test_examples = [examples[i] for i in indexs[train_size + valid_size:train_size + valid_size + test_size]]

if adj_tf_threshold > 0:
    adj_tf.data *= (adj_tf.data > adj_tf_threshold)
    adj_tf.eliminate_zeros()
if adj_mi_threshold > 0:
    adj_mi.data *= (adj_mi.data > adj_mi_threshold)
    adj_mi.eliminate_zeros()
if adj_kg_threshold > 0:
    adj_kg.data *= (adj_kg.data > adj_kg_threshold)
    adj_kg.eliminate_zeros()

lambda_t = args.tf_scale
lambda_p = args.mi_scale
lambda_k = args.kg_scale
adj_tf *= lambda_t
adj_mi *= lambda_p
adj_kg *= lambda_k
adj_tf.setdiag(1.0)
adj_mi.setdiag(1.0)
adj_kg.setdiag(1.0)

mg_list = []
mg_names = ''
if args.graph_mode == '0':
    mg_names = 'none'
else:
    if '1' in args.graph_mode:
        mg_names += 'tf'
        mg_list += [adj_tf]
    if '2' in args.graph_mode:
        if len(mg_names) > 0:
            mg_names += '+'
        mg_names += 'mi'
        mg_list += [adj_mi]
    if '3' in args.graph_mode:
        if len(mg_names) > 0:
            mg_names += '+'
        mg_names += 'kg'
        mg_list += [adj_kg]

norm_mg_list = []
for i in range(len(mg_list)):
    adj = mg_list[i]
    adj = normalize_adj(adj)
    norm_mg_list.append(sparse_scipy2torch(adj.tocoo()).to(device))
mg_list = norm_mg_list

del adj_tf, adj_mi, adj_kg
gc.collect()

tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)


def get_pytorch_dataloader(examples, tokenizer, batch_size):
    ds = CorpusDataset(examples, tokenizer, vocab_map, vocab_bert_ids_map, max_len)
    return ds.id2bert, DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=ds.pad)


dataset_bert_ids, _ = get_pytorch_dataloader(train_examples, tokenizer, batch_size)

model_file_load = '{}_model_{}_graph[{}]_run{}_dyn{}_max_epoch{}.pt'.format(model_name, dataset, mg_names, run,
                                                                            lambda_dyn, args.epoch)
checkpoint = torch.load(os.path.join(output_dir, model_file_load), map_location='cpu')
if 'step' in checkpoint:
    prev_save_step = checkpoint['step']
    start_epoch = checkpoint['epoch']
else:
    prev_save_step = -1
    start_epoch = checkpoint['epoch'] + 1
model = MGBERT_MLM.from_pretrained(bert_model_scale, state_dict=checkpoint['model_state'], adj_dim=vocab_size,
                                   adj_num=len(mg_list), dyn=lambda_dyn, bert_ids=dataset_bert_ids)
print('Loaded the pretrain model:', model_file_load, ', epoch:', checkpoint['epoch'], 'step:', prev_save_step,
      'valid acc:', checkpoint['valid_acc'])
model.to(device)


def predict(model, examples, tokenizer, batch_size):
    dataset_bert_ids, dataloader = get_pytorch_dataloader(examples, tokenizer, batch_size)
    predict_out = []
    true_out = []
    masked_examples = []
    masked_probs = []
    model.eval()
    ks = [1, 5]
    all_hits = {k: 0 for k in ks}
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, swop_eye, true_input_ids, just_masked_input_index = batch
            prediction_scores = model(mg_list, swop_eye, input_ids, segment_ids, input_mask)
            just_masked_input_index = just_masked_input_index.flatten()
            masked_tokens_scores = prediction_scores[range(len(prediction_scores)), just_masked_input_index]
            masked_tokens_true_ids = true_input_ids[range(len(prediction_scores)), just_masked_input_index]
            masked_probs += list(
                np.amax(torch.nn.functional.softmax(masked_tokens_scores, dim=1).cpu().numpy(), axis=1))

            hits = hits_at_k(masked_tokens_scores, masked_tokens_true_ids, ks)

            input_ids = input_ids.cpu().numpy()
            masked_tokens_true_ids = masked_tokens_true_ids.cpu().numpy()
            for j in range(len(input_ids)):
                masked_examples.append(tokenizer.convert_ids_to_tokens(input_ids[j]))
                _, max_idx = torch.topk(masked_tokens_scores[j], k=5)
                max_idx = max_idx.cpu().numpy()
                predict_out.append(tokenizer.convert_ids_to_tokens(max_idx))
                true_out.append(tokenizer.convert_ids_to_tokens([masked_tokens_true_ids[j]]))

            for k, hit in hits.items():
                all_hits[k] += hit
            total += len(prediction_scores)

        for k, hit in all_hits.items():
            all_hits[k] = hit / total
        perplexity = np.power(2, -np.sum(np.log2(np.array(masked_probs))) / len(masked_probs))
    return masked_examples, predict_out, true_out, all_hits, perplexity


f_name = 'results/evaluated/{}_model_{}_graph[{}]_run{}_gcn{}_max_epoch{}.txt'.format(model_name, dataset, mg_names,
                                                                                      run, lambda_dyn, args.epoch)
with open(f_name, 'a+') as f:
    if args.data == 'valid':
        exs = valid_examples
    elif args.data == 'test':
        exs = test_examples
    elif args.data == 'train':
        exs = train_examples
    f.write('Results on ' + args.data + ' set\n' + '-' * 50 + '\n\n')
    all_hits = {}
    perplexities = []
    for _ in range(5):
        masked_examples, predicted, trues, hits, prplx = predict(model, exs, tokenizer, batch_size)
        perplexities.append(prplx)
        for k, hit in hits.items():
            if k in all_hits:
                all_hits[k].append(hit.cpu().numpy())
            else:
                all_hits[k] = [hit.cpu().numpy()]
    all_hits_mean, all_hits_std = {}, {}
    for k, hits in all_hits.items():
        print(k, ':', hits)
        all_hits_mean[k] = np.mean(hits)
        all_hits_std[k] = np.std(hits)

    f.write('MEAN: ' + get_hits_str(all_hits_mean) + '\n')
    f.write('STD: ' + get_hits_str(all_hits_std) + '\n')
    f.write('Perplexity: {:.3f}. +- {:.3f}\n\n'.format(np.mean(perplexities), np.std(perplexities)))
    print('MEAN', get_hits_str(all_hits_mean), '\tSTD', get_hits_str(all_hits_std),
          '\nPerplexity: {:.3f}. +- {:.3f}'.format(np.mean(perplexities), np.std(perplexities)))

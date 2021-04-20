# -*- coding: utf-8 -*-

import argparse
import gc
import os
import pickle as pkl
import time

from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import *

random.seed(int(time.time()))
np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))

cuda_available = torch.cuda.is_available()
# cuda_available = False
if cuda_available:
    torch.cuda.manual_seed_all(int(time.time()))
device = torch.device("cuda:0" if cuda_available else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cola')
parser.add_argument('--kg', type=str, default='WN18')
parser.add_argument('--just-masked', type=int, default=1)  # loss on just masked tokens
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--graph-mode', type=str, default='0')  # 123 : tf-mi-kg
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--sw', type=int, default='0')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--dyn', type=float, default=0)  # dyn = 0: static, 0 < dyn < 1: dynamic
parser.add_argument('--lr', type=float, default=0)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--model', type=str, default='MG-BERT')

parser.add_argument('--kg-scale', type=float, default=1)
parser.add_argument('--tf-scale', type=float, default=1)
parser.add_argument('--mi-scale', type=float, default=1)

parser.add_argument('--batch', type=int, default=64)

'''
Load configs
'''
args = parser.parse_args()
run = args.run
loss_on_just_masked = True if args.just_masked == 1 else False

batch_size = args.batch

dataset = args.dataset
kg = args.kg
model_name = args.model
stop_words = True if args.sw == 1 else False
load_from_checkpoint = True if args.load == 1 else False

lambda_dyn = args.dyn
l2_decay = args.l2

dataset_list = {'sst-2', 'cola', 'brown'}
folder_names = {'cola': 'CoLA', 'sst-2': 'SST-2', 'brown': 'Brown'}

total_train_epochs = args.epoch
report_each_epoch = 0  # save loss figure every #n epochs
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
gradient_accumulation_steps = 1
bert_model_scale = 'bert-base-uncased'
do_lower_case = True
warmup_proportion = 0.1

data_dir = 'data/dump'
output_dir = 'output/'
task = 'MLM'
adj_mi_threshold = 0.2
adj_tf_threshold = 0
adj_kg_threshold = 0

print(model_name + ' Start at:', time.asctime())
print('\n----- Configure -----',
      '\n  dataset:', dataset,
      '\n  stop_words:', stop_words,
      '\n  Learning_rate0:', learning_rate0, 'weight_decay:', l2_decay,
      '\n  Max Length:', max_len,
      '\n  Knowledge Graph:', kg,
      '\n  loss on just masked tokens:', loss_on_just_masked)

'''
Load dataset and multi-graph
'''
print('\n----- Prepare data set -----')
print('  Load/shuffle/seperate', dataset, 'dataset, and vocabulary graph adjacent matrix')

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

# hyperparameters of the importance of each graph
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

print_file_args = {'ds': dataset, 'loss_masked': args.just_masked, 'run': run, 'graph_mode': mg_names,
                   'model': model_name, 'dyn': lambda_dyn}
model_file_save = '{}_model_{}_graph[{}]_run{}_dyn{}'.format(model_name, dataset, mg_names, run, lambda_dyn)
print('model_file_save:', model_file_save, '\n  graph mode:', mg_names)

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
    return ds.id2bert, DataLoader(dataset=ds,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=ds.pad)


dataset_bert_ids, train_dataloader = get_pytorch_dataloader(train_examples, tokenizer, batch_size)
_, valid_dataloader = get_pytorch_dataloader(valid_examples, tokenizer, batch_size)
_, test_dataloader = get_pytorch_dataloader(test_examples, tokenizer, batch_size)

total_train_steps = int(len(train_dataloader) / gradient_accumulation_steps * total_train_epochs)

print('  Num examples for train =', len(train_dataloader) * batch_size)
print("  Num examples for validate = %d" % len(valid_examples))
print("  Batch size = %d" % batch_size)
print("  Num steps = %d" % total_train_steps)


def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name, ks=(1, 5)):
    model.eval()
    total = 0
    ev_loss = 0
    all_hits = {k: 0 for k in ks}
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, swop_eye, true_input_ids, just_masked_input_index = batch
            prediction_scores = model(mg_list, swop_eye, input_ids, segment_ids, input_mask)

            just_masked_input_index = just_masked_input_index.flatten()
            masked_tokens_scores = prediction_scores[range(len(prediction_scores)), just_masked_input_index]
            masked_tokens_true_ids = true_input_ids[range(len(prediction_scores)), just_masked_input_index]
            if loss_on_just_masked:
                loss = loss_ent(masked_tokens_scores.view(-1, model.config.vocab_size), masked_tokens_true_ids.view(-1))
            else:
                loss = loss_ent(prediction_scores.view(-1, model.config.vocab_size), true_input_ids.view(-1))
            ev_loss += loss.item()

            hits = hits_at_k(masked_tokens_scores, masked_tokens_true_ids, ks)  # by default Hits@1 and Hits@5
            for k, hit in hits.items():
                all_hits[k] += hit
            total += len(prediction_scores)

    for k, hit in all_hits.items():
        all_hits[k] = hit / total
    end = time.time()
    s = 'Epoch : %d %s on %s, Spend:%.3f minutes for evaluation' % (
        epoch_th, get_hits_str(all_hits), dataset_name, (end - start) / 60.0)
    print_and_write(s, print_file_args)
    print_and_write('-' * 70, print_file_args)
    return ev_loss, all_hits


'''
Load or create model
'''
from model import MGBERT_MLM

print("\n----- Running training -----")
model_file_load = 'MG-BERT_model_cola_graph[tf+mi+kg]_run1_dyn0.2_max_epoch100.pt'
if load_from_checkpoint and os.path.exists(os.path.join(output_dir, model_file_load)):
    checkpoint = torch.load(os.path.join(output_dir, model_file_load), map_location='cpu')

    if 'step' in checkpoint:
        prev_save_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
    else:
        prev_save_step = -1
        start_epoch = checkpoint['epoch'] + 1
    valid_acc_prev = checkpoint['valid_acc']
    model = MGBERT_MLM.from_pretrained(bert_model_scale, state_dict=checkpoint['model_state'],
                                       adj_dim=vocab_size, adj_num=len(mg_list), dyn=lambda_dyn,
                                       bert_ids=dataset_bert_ids)
    pretrained_dict = checkpoint['model_state']
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    print('Loaded the pretrain model:', model_file_load, ', epoch:', checkpoint['epoch'], 'step:', prev_save_step,
          'valid acc:', checkpoint['valid_acc'])

else:
    start_epoch = 0
    valid_acc_prev = 0
    model = MGBERT_MLM.from_pretrained(bert_model_scale, adj_dim=vocab_size, adj_num=len(mg_list), dyn=lambda_dyn,
                                       bert_ids=dataset_bert_ids)
    prev_save_step = -1

model.to(device)

'''
Training phase
'''
optimizer = BertAdam(model.parameters(), lr=learning_rate0, warmup=warmup_proportion, t_total=total_train_steps,
                     weight_decay=l2_decay)

loss_ent = CrossEntropyLoss()

train_start = time.time()
global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)

all_loss_list = {'train': [], 'valid': [], 'test': []}
all_hits_list = {'train': [], 'valid': [], 'test': []}
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    ep_train_start = time.time()
    model.train()
    optimizer.zero_grad()

    max_epoch = (((epoch // 100) + 1) * 100)  # to save the model every 100 epochs
    if max_epoch > total_train_epochs:
        max_epoch = total_train_epochs
    print_file_args['max_epoch'] = max_epoch
    if epoch == start_epoch:
        print_and_write(str(args) + '\n', print_file_args)

    for step, batch in enumerate(train_dataloader):
        if prev_save_step > -1:
            if step <= prev_save_step: continue
        if prev_save_step > -1:
            prev_save_step = -1
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, swop_eye, true_input_ids, just_masked_input_index = batch
        prediction_scores = model(mg_list, swop_eye, input_ids, segment_ids, input_mask)

        if loss_on_just_masked:  # as in BERT
            just_masked_input_index = just_masked_input_index.flatten()
            loss = loss_ent(prediction_scores[range(len(prediction_scores)), just_masked_input_index].view(-1,
                                                                                                           model.config.vocab_size),
                            true_input_ids[range(len(prediction_scores)), just_masked_input_index].view(-1))
        else:
            loss = loss_ent(prediction_scores.view(-1, model.config.vocab_size), true_input_ids.view(-1))

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()

        tr_loss += loss.item()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1
        if step % 40 == 0:
            s = "Epoch:{}-{}/{}, Train Loss: {}, Cumulated time: {}m ".format(epoch, step, len(train_dataloader),
                                                                              loss.item(),
                                                                              (time.time() - train_start) / 60.0)
            print_and_write(s, print_file_args)

    print_and_write('-' * 70, print_file_args)
    valid_loss, valid_hits = evaluate(model, valid_dataloader, batch_size, epoch, 'Valid_set')
    test_loss, test_hits = evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')
    valid_acc = valid_hits[1]
    test_acc = test_hits[1]
    all_loss_list['train'].append(tr_loss)
    all_loss_list['valid'].append(valid_loss)
    all_loss_list['test'].append(test_loss)
    all_hits_list['valid'].append(valid_hits)
    all_hits_list['test'].append(test_hits)

    print_and_write(
        "Epoch:{} completed, Total Train Loss:{}, Valid Loss:{}, Spend {}m ".format(epoch, tr_loss, valid_loss,
                                                                                    (time.time() - train_start) / 60.0),
        print_file_args)
    print_and_write('-' * 70, print_file_args)

    # Save a checkpoint when a better model is obtained
    if valid_acc > valid_acc_prev:
        to_save = {'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
                   'lower_case': do_lower_case}
        torch.save(to_save, os.path.join(output_dir, model_file_save + '_max_epoch{}.pt'.format(max_epoch)))
        valid_acc_prev = valid_acc
        test_acc_when_valid_best = test_acc
        test_hits_when_valid_best = test_hits
        valid_acc_best = valid_acc
        valid_hits_best = valid_hits
        valid_acc_best_epoch = epoch
    if (epoch + 1) % report_each_epoch == 0:
        savefig(all_loss_list, print_file_args)

print_and_write("\n**Optimization Finished!,Total spend: %.3f" % ((time.time() - train_start) / 60.0), print_file_args)
print_and_write("**Valid %s at %d epoch." % (get_hits_str(valid_hits_best), valid_acc_best_epoch), print_file_args)
print_and_write("**Test hits when valid best: %s" % (get_hits_str(test_hits_when_valid_best)), print_file_args)

savefig(all_loss_list, print_file_args)

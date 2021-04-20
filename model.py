# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init

from pytorch_pretrained_bert.modeling import BertEmbeddings, BertEncoder, BertLayer, BertModel, BertOnlyMLMHead


class SPGCN_Dynamic(nn.Module):
    def __init__(self, config, voc_dim, emb_dim=768, dropout_rate=0.2, num_adj=1, lambda_dyn=0):
        super(SPGCN_Dynamic, self).__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.emb_dim = emb_dim
        self.lambda_dyn = lambda_dyn
        self.config = config
        self.sent_layer = BertLayer(config, output_attentions=False, keep_multihead_output=False)
        for i in range(self.num_adj):
            setattr(self, 'W%d_vh' % i, nn.Parameter(torch.randn(emb_dim, emb_dim)))
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith('W') or n.startswith('a') or n in ('W', 'a', 'dense'):
                init.eye_(p)

    def forward(self, vocab_adj_list, X_dv, swop_eye, attention_mask=None):
        for i in range(self.num_adj):
            h = X_dv.mm(getattr(self, 'W%d_vh' % i))
            H_dh = torch.mm(vocab_adj_list[i], h)
            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        sent = self.sent_layer(swop_eye.matmul(X_dv), attention_mask)
        H_prime = swop_eye.matmul(fused_H)
        H_prime = (1 - self.lambda_dyn) * H_prime + self.lambda_dyn * sent
        return H_prime


class GraphEmbeddings(BertEmbeddings):

    def __init__(self, config, adj_dim, adj_num, dyn, bert_ids):
        super(GraphEmbeddings, self).__init__(config)
        self.vocab_gcn = SPGCN_Dynamic(config, voc_dim=768, num_adj=adj_num,
                                       lambda_dyn=dyn)  # dyn = 0: static, 0 < dyn < 1: dynamic
        self.dyn = dyn
        self.bert_ids = bert_ids

    def forward(self, vocab_adj_list, swop_eye, input_ids, token_type_ids=None):
        words_embeddings = self.word_embeddings(input_ids)
        all_words_embeddings = self.word_embeddings(self.bert_ids.to(input_ids.device))

        if self.dyn < 1:
            attention_mask = torch.ones_like(input_ids)
            extended_attention_mask = attention_mask.reshape(attention_mask.size(0), 1, 1, -1)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            gcn_words_embeddings = self.vocab_gcn(vocab_adj_list, all_words_embeddings, swop_eye,
                                                  attention_mask=extended_attention_mask)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = gcn_words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MGBERT(BertModel):
    def __init__(self, config, adj_dim, adj_num, output_attentions=False,
                 keep_multihead_output=False, dyn=0, bert_ids=None):
        super(MGBERT, self).__init__(config, output_attentions, keep_multihead_output)
        self.dyn = dyn
        self.bert_ids = bert_ids
        self.output_attentions = output_attentions
        self.embeddings = GraphEmbeddings(config, adj_dim, adj_num, dyn, bert_ids)
        self.encoder = BertEncoder(config, output_attentions=output_attentions,
                                   keep_multihead_output=keep_multihead_output)
        self.apply(self.init_bert_weights)

    def forward(self, vocab_adj_list, swop_eye, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, head_mask=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(vocab_adj_list, swop_eye, input_ids, token_type_ids)
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        output_all_encoded_layers = True if self.output_attentions else False
        encoded_layers = self.encoder(embedding_output, extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers, head_mask=head_mask)
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers

        sequence_output = encoded_layers[-1]
        return sequence_output


class MGBERT_MLM(BertModel):
    def __init__(self, config, adj_dim, adj_num, output_attentions=False,
                 keep_multihead_output=False, dyn=0, bert_ids=None):
        super(MGBERT_MLM, self).__init__(config, output_attentions, keep_multihead_output)
        self.mgbert = MGBERT(config, adj_dim, adj_num, output_attentions=output_attentions,
                             keep_multihead_output=keep_multihead_output, dyn=dyn, bert_ids=bert_ids)
        self.cls = BertOnlyMLMHead(config, self.mgbert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, vocab_adj_list, swop_eye, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, head_mask=None):
        sequence_output = self.mgbert(vocab_adj_list, swop_eye, input_ids, token_type_ids=token_type_ids,
                                      attention_mask=attention_mask, masked_lm_labels=masked_lm_labels,
                                      head_mask=head_mask)
        prediction_scores = self.cls(sequence_output)
        return prediction_scores

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from allennlp.modules.elmo import Elmo, batch_to_ids


CUDA = torch.cuda.is_available()
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0.0, requires_grad=False)
if CUDA:
    elmo = elmo.cuda()


def get_word_embeddings(sentence):
    character_ids = batch_to_ids(sentence)
    if CUDA:
        character_ids = character_ids.cuda()
    embedding = elmo(character_ids)
    outp_ctxt = embedding['elmo_representations'][0]
    ctxt_mask = embedding['mask']
    return outp_ctxt, ctxt_mask


class BiLSTM(nn.Module):

    def __init__(self, config, is_pos=False):
        super(BiLSTM, self).__init__()
        self.bidirectional = config['bidirectional']
        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.embedding_dim = config['embedding_dim']

        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim, config['num_layers'],
                              batch_first=True, bidirectional=config['bidirectional']) #, dropout=config['dropout']

    def init_weights(self):
        for name, param in self.bilstm.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)

    def forward(self, emb, len_inp, hidden=None):
        len_inp = len_inp.cpu().numpy() if CUDA else len_inp.numpy()
        len_inp, idx_sort = np.sort(len_inp)[::-1], np.argsort(-len_inp)
        len_inp = len_inp.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if CUDA else torch.from_numpy(idx_sort)
        emb = emb.index_select(0, Variable(idx_sort))

        emb_packed = pack_padded_sequence(emb, len_inp, batch_first=True)
        outp, _ = self.bilstm(emb_packed, hidden)
        outp = pad_packed_sequence(outp, batch_first=True)[0]

        idx_unsort = torch.from_numpy(idx_unsort).cuda() if CUDA else torch.from_numpy(idx_unsort)
        outp = outp.index_select(0, Variable(idx_unsort))
        return outp

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        num_directions = 2 if self.bidirectional else 1
        return (Variable(weight.new(self.num_layers*num_directions, batch_size, self.hidden_dim).zero_()),
                Variable(weight.new(self.num_layers*num_directions, batch_size, self.hidden_dim).zero_()))


#  Returns LSTM based sentence encodin, dim=1024, elements of vector in range [-1,1]
class SentenceEncoder(nn.Module):

    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        self.context_encoder = BiLSTM(config)
        self.inner_pred = nn.Linear((config['hidden_dim']*2), config['hidden_dim']*2) # Prafulla 3
        self.ws1 = nn.Linear((config['hidden_dim']*2), (config['hidden_dim']*2))
        self.ws2 = nn.Linear((config['hidden_dim']*2), 1)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(config['dropout'])

    def init_weights(self):
        nn.init.xavier_uniform(self.ws1.state_dict()['weight'])
        self.ws1.bias.data.fill_(0)
        nn.init.xavier_uniform(self.ws2.state_dict()['weight'])
        self.ws2.bias.data.fill_(0)
        nn.init.xavier_uniform(self.inner_pred.state_dict()['weight'])
        self.inner_pred.bias.data.fill_(0)
        self.context_encoder.init_weights()

    def forward(self, outp_ctxt, ctxt_mask, length, hidden_ctxt=None):
        outp = self.context_encoder.forward(outp_ctxt, length, hidden_ctxt)

        self_attention = F.tanh(self.ws1(self.drop(outp)))
        self_attention = self.ws2(self.drop(self_attention)).squeeze()
        self_attention = self_attention + -10000*(ctxt_mask == 0).float()
        self_attention = self.drop(self.softmax(self_attention))
        sent_encoding = torch.sum(outp*self_attention.unsqueeze(-1), dim=1)

        return F.tanh(self.inner_pred(self.drop(sent_encoding)))


#  Returns Transition scores, note it ignores headline and the first sentence
#  Output size Number of sentences-1 * 2, start from the second sentence
class TopicSegmenter(nn.Module):

    def __init__(self, config):
        super(TopicSegmenter, self).__init__()
        self.W1 = nn.Linear(config['hidden_dim']*2, config['hidden_dim']*2, bias=False)
        self.W2 = nn.Linear(config['hidden_dim']*2, config['hidden_dim']*2, bias=False)
        self.vt = nn.Linear(config['hidden_dim']*4, 1)
        self.decoding_rnn = nn.LSTMCell(input_size=config['hidden_dim']*2, hidden_size=config['hidden_dim']*2)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(config['dropout'])

    def init_weights(self):
        nn.init.xavier_uniform(self.W1.state_dict()['weight'])
        nn.init.xavier_uniform(self.W2.state_dict()['weight'])
        nn.init.xavier_uniform(self.vt.state_dict()['weight'])
        for name, param in self.decoding_rnn.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        self.vt.bias.data.fill_(0)

    def attentions(self, decoder_step, encoder_outputs, mask):
        e1 = self.W1(self.drop(encoder_outputs.squeeze()))
        e2 = self.W2(self.drop(decoder_step))
        enc = torch.cat([e1*e2, e1-e2],1)
        attn = self.vt(self.drop(torch.tanh(enc))) + -10000. * (mask == 0).float()
        return attn.squeeze()

    def get_sample(self, attn, is_test):
        scores = F.softmax(attn)
        if is_test:
            _, predicts = torch.max(scores, 0)
        else:
            predicts = Categorical(scores).sample()
        return predicts.item()

    def forward(self, sent_encodings, hidden_embedding, is_test):
        sent_to_topic = [1]
        rl_loss = None
        sent_encodings = sent_encodings.transpose(0,1)
        maxlen = sent_encodings.size(0) # No of sentences, headline, end sentence pad
        decoder_hidden = hidden_embedding
        while True:  # ignore last sentence as maxpool will always pick that sentence during inference
            mask = [[1] for _ in range(maxlen)]
            for i in range(sent_to_topic[-1]+1): # index and sentence numbers are aligned, so +1
                mask[i][0] = 0
            mask = torch.LongTensor(mask).cuda()
            decoder_input = sent_encodings[sent_to_topic[-1]]
            h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)
            decoder_hidden = (h_i, c_i)
            scores = self.attentions(h_i, sent_encodings, mask)
            ind = self.get_sample(scores, is_test)
            if len(sent_to_topic) == 1: # added the first transition sentence other than the first sentence
                rl_loss = F.cross_entropy(scores.view(1, -1), torch.LongTensor([ind]).cuda())
            else:
                # ignore previous scores, keep scores following last transition
                # print(scores[sent_to_topic[-1]+1:].view(1, -1), ind-sent_to_topic[-1]-1)
                rl_loss += F.cross_entropy(scores[sent_to_topic[-1]+1:].view(1, -1), torch.LongTensor([ind-sent_to_topic[-1]-1]).cuda())
                # print(rl_loss)
            sent_to_topic.append(ind)  # first headline, so index and sentence numbers are aligned
            if ind == maxlen-1:  # last sentence selected, stop
                break
        return sent_to_topic, rl_loss/(len(sent_to_topic)-1+1e-5)

    def rule_based_scores(self, sent_encodings, hidden_embedding, rule_based_trans_sents):
        sent_to_topic = [1]
        rl_loss = None
        sent_encodings = sent_encodings.transpose(0,1)
        maxlen = sent_encodings.size(0)
        assert maxlen == len(rule_based_trans_sents)+1  # sent_encodings include headline
        decoder_hidden = hidden_embedding
        for ind, elem in enumerate(rule_based_trans_sents):  # sentence number = ind+1
            if ind == 0:
                continue
            if elem == 1:
                mask = [[1] for _ in range(maxlen)]
                for i in range(sent_to_topic[-1]+1): # index in sent_encodings and sentence numbers are aligned, so +1
                    mask[i][0] = 0
                mask = torch.LongTensor(mask).cuda()
                decoder_input = sent_encodings[sent_to_topic[-1]] # sentence numbers are 1 indexed in sent_encodings, since headline is included
                h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)
                decoder_hidden = (h_i, c_i)
                scores = self.attentions(h_i, sent_encodings, mask)
                if len(sent_to_topic) == 1:  # added the first transition sentence other than the first sentence
                    rl_loss = F.cross_entropy(scores.view(1, -1), torch.LongTensor([ind+1]).cuda())
                else:
                    rl_loss += F.cross_entropy(scores[sent_to_topic[-1]+1:].view(1, -1), torch.LongTensor([ind-sent_to_topic[-1]]).cuda())  # ind + 1 = sentence number
                sent_to_topic.append(ind + 1)
        # print("----",sent_to_topic)
        return sent_to_topic, rl_loss/(len(sent_to_topic)-1+1e-5) if rl_loss is not None else None


class DiscourseEncoder(nn.Module):

    def __init__(self, config):
        super(DiscourseEncoder, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.ws3 = nn.Linear((config['hidden_dim']*2), (config['hidden_dim']*2))
        self.ws4 = nn.Linear((config['hidden_dim']*2), 1)
        self.softmax = nn.Softmax(dim=1)
        self.discourse_encoder = nn.LSTM(config['hidden_dim']*2, config['hidden_dim'], config['num_layers'],
                              batch_first=True, bidirectional=config['bidirectional'])

    def init_weights(self):
        for name, param in self.discourse_encoder.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)

        nn.init.xavier_uniform(self.ws3.state_dict()['weight'])
        self.ws3.bias.data.fill_(0)
        nn.init.xavier_uniform(self.ws4.state_dict()['weight'])
        self.ws4.bias.data.fill_(0)

    def forward(self, sent_encoding, hidden_ctxt=None):
        inner_pred = self.drop(sent_encoding)
        inner_pred, hidden_op = self.discourse_encoder.forward(inner_pred[None, :, :])
        self_attention = F.tanh(self.ws3(self.drop(inner_pred)))
        self_attention = self.ws4(self.drop(self_attention)).squeeze(2)
        doc_self_attention = self.softmax(self_attention)
        doc_inner_pred = inner_pred.squeeze()
        doc_disc_encoding = torch.sum(doc_inner_pred * doc_self_attention.unsqueeze(-1), dim=1)
        return inner_pred, self_attention, hidden_op, doc_disc_encoding


class DiscourseProfiler(nn.Module):

    def __init__(self, config):
        super(DiscourseProfiler, self).__init__()
        self.pre_pred = nn.Linear((config['hidden_dim']*2*5), config['hidden_dim']*2)
        self.pred = nn.Linear((config['hidden_dim']*2), config['out_dim'])
        self.drop = nn.Dropout(config['dropout'])
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        nn.init.xavier_uniform(self.pre_pred.state_dict()['weight'])
        self.pre_pred.bias.data.fill_(0)
        nn.init.xavier_uniform(self.pred.state_dict()['weight'])
        self.pred.bias.data.fill_(0)

    def get_du_representations(self, sent_encoding, self_attention, trans_sents, doc_disc_encoding):
        out_1 = []
        seq_len = sent_encoding.size(1)

        for i in range(len(trans_sents)):
            mask = [0] * seq_len
            if i + 1 < len(trans_sents):
                last_index = trans_sents[i + 1]
            else:
                last_index = seq_len
            # print(last_index)
            # for index in range(trans_sents[i], trans_sents[i]+1):
            for index in range(trans_sents[i], last_index, 1):
                mask[index] = 1
            mask = torch.LongTensor([mask]).cuda()

            _self_attention = self_attention + -10000 * (mask == 0).float()
            _self_attention = self.softmax(_self_attention)
            _inner_pred = sent_encoding[:, trans_sents[i]:last_index, :]
            disc_encoding = torch.sum(sent_encoding * _self_attention.unsqueeze(-1), dim=1)
            _inner_pred = _inner_pred.squeeze(0)
            resize_doc_disc_encoding = doc_disc_encoding.expand(_inner_pred.size())

            out_1.append(torch.cat([_inner_pred, _inner_pred * disc_encoding, _inner_pred - disc_encoding,
                                    resize_doc_disc_encoding * disc_encoding, resize_doc_disc_encoding - disc_encoding], 1))

        return torch.cat(out_1, 0)

    def forward(self, sent_encoding, self_attention, trans_sents, hidden_ctxt=None):
        doc_self_attention = self.softmax(self_attention)
        doc_inner_pred = sent_encoding.squeeze()
        doc_disc_encoding = torch.sum(doc_inner_pred * doc_self_attention.unsqueeze(-1), dim=1)
        out = self.get_du_representations(sent_encoding, self_attention, trans_sents, doc_disc_encoding)
        pre_pred = F.tanh(self.pre_pred(self.drop(out)))
        return self.pred(self.drop(pre_pred))


class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.discourse_encoder = DiscourseEncoder(config)
        self.discourse_profiler = DiscourseProfiler(config)
        self.topic_segmenter = TopicSegmenter(config)
        self.cross_entropy = nn.CrossEntropyLoss()
        # self.loss_weights = nn.Parameter(torch.ones(2))
        # print(self.loss_weights)

    def init_weights(self):
        self.sentence_encoder.init_weights()
        self.discourse_encoder.init_weights()
        self.discourse_profiler.init_weights()
        self.topic_segmenter.init_weights()

    def get_topic_transition_scores(self, sent_encoding, hidden_embedding, is_test):
        return self.topic_segmenter.forward(sent_encoding, hidden_embedding, is_test)

    def get_rule_transition_scores(self, sent_encoding, hidden_embedding, rule_based_trans_sents):
        return self.topic_segmenter.rule_based_scores(sent_encoding, hidden_embedding, rule_based_trans_sents)

    def get_sentence_encoding(self, outp_ctxt, ctxt_mask, length, hidden_ctxt=None):
        return self.sentence_encoder.forward(outp_ctxt, ctxt_mask, length, hidden_ctxt)

    def get_discourse_encoding(self, sent_encoding):
        return self.discourse_encoder.forward(sent_encoding)

    def get_discourse_tags(self, sent_encoding, self_attention, trans_sents):
        return self.discourse_profiler.forward(sent_encoding, self_attention, trans_sents)

    def forward(self, sentence, length, rule_based_trans_sents=None, tt_trans_sents=None, is_test=False, hidden_ctxt=None):
        ground_pred, dp_loss, op_tt, tt_loss = None, None, None, None
        outp_ctxt, ctxt_mask = get_word_embeddings(sentence)
        sent_encoding = self.get_sentence_encoding(outp_ctxt, ctxt_mask, length)
        sent_encoding, self_attention, hidden_enc, doc_encoding = self.get_discourse_encoding(sent_encoding)
        hidden_op = (doc_encoding.view(1,-1), doc_encoding.view(1,-1))
        trans_sents, rl_loss = self.get_topic_transition_scores(sent_encoding, hidden_op, is_test)

        pred = self.get_discourse_tags(sent_encoding, self_attention, trans_sents)

        if rule_based_trans_sents is not None:
            # For self critic, use is_test=True in get_transitions_sample.
            rule_trans_sents, dp_loss = self.get_rule_transition_scores(sent_encoding, hidden_op, rule_based_trans_sents)  # For baseline
            # rule_trans_sents, _ = self.get_topic_transition_scores(sent_encoding, hidden_op, True)   # For self-critic
            ground_pred = self.get_discourse_tags(sent_encoding, self_attention, rule_trans_sents)

        if tt_trans_sents is not None:
            tt_trans_sents, tt_loss = self.get_rule_transition_scores(sent_encoding, hidden_op, tt_trans_sents)
            op_tt = self.get_discourse_tags(sent_encoding, self_attention, tt_trans_sents)
        return pred, ground_pred, op_tt, rl_loss, dp_loss, tt_loss, trans_sents  # F.softmax(self.loss_weights)

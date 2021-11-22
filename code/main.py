import os
from process_file import process_doc
import random
import torch
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from model import Classifier
import torch.nn as nn
import torch.optim as optim
import time


def get_batch(doc, ref_type='headline'):
    sent, ls, out, sids, transitions = [], [], [], [], []
    transitions_tt = []
    sent.append(doc.headline)
    ls.append(len(doc.headline))
    for sid in doc.sids:
        # print(sid)
        if SPEECH:
            out.append(out_map[doc.sent_to_speech.get(sid, 'NA')])
        else:
            out.append(out_map[doc.sent_to_event.get(sid, 'NA')])

        sent.append(doc.sentences[sid])
        ls.append(len(doc.sentences[sid]))
        sids.append(sid)
        transitions.append(doc.sent_to_topic[sid])
        transitions_tt.append(doc.sent_to_topic_text_tile[sid])

    # Adding EOD
    sent.append(["eod"])
    ls.append(1)
    transitions.append(1)
    transitions_tt.append(1)
    ls = torch.LongTensor(ls)
    out = torch.LongTensor(out)
    assert len(transitions) == len(sent)-1
    assert len(transitions_tt) == len(sent)-1
    return sent, ls, out, sids, transitions, transitions_tt  # transitions[1:]


def train(epoch, data, seed):
    random.shuffle(data)
    start_time = time.time()
    profiler_loss = 0
    transition_cross_entropy_loss = 0
    transition_rl_loss = 0
    supervise_cnt = [0, 0]
    optimizer.zero_grad()
    global prev_best_macro

    for ind, doc in enumerate(data):
        model.train()
        # if doc.name in ['Anyt_corpus_data_2007_01_09_1817645.txt']:
        #     continue
        sent, ls, out, sids, trans_sents, trasn_sents_tt = get_batch(doc)
        y_true = list(out.numpy())

        if has_cuda:
            ls = ls.cuda()
            out = out.cuda()

        _output, ground_output, op_tt, rl_loss, dp_loss, tt_loss, transition_sentences = \
            model.forward(sent, ls, rule_based_trans_sents=trans_sents, tt_trans_sents=trasn_sents_tt)

        _output = _output[:-1]
        temp_loss = criterion(_output, out)  # last sentence is eod
        profiler_loss += temp_loss.item()
        loss = temp_loss

        _output = _output.squeeze()
        _, predict = torch.max(_output, 1)
        y_pred = list(predict.cpu().detach().numpy() if has_cuda else predict.detach().numpy())
        reward = precision_recall_fscore_support(y_true, y_pred, average='macro')[2] + \
                 precision_recall_fscore_support(y_true, y_pred, average='micro')[2]

        ground_output = ground_output.squeeze()
        ground_output = ground_output[:-1]
        _, ground_predict = torch.max(ground_output, 1)
        y_pred_ground = list(ground_predict.cpu().detach().numpy() if has_cuda else ground_predict.detach().numpy())
        reward_ip = precision_recall_fscore_support(y_true, y_pred_ground, average='micro')[2] + \
                 precision_recall_fscore_support(y_true, y_pred_ground, average='macro')[2]

        op_tt = op_tt.squeeze()
        op_tt = op_tt[:-1]
        _, op_tt_predict = torch.max(op_tt, 1)
        y_op_tt = list(op_tt_predict.cpu().detach().numpy() if has_cuda else op_tt_predict.detach().numpy())
        reward_tt = precision_recall_fscore_support(y_true, y_op_tt, average='macro')[2] + \
                 precision_recall_fscore_support(y_true, y_op_tt, average='micro')[2]

        reward_list = [reward.item(0), reward_ip.item(0), reward_tt.item(0)]
        idxs = sorted(range(len(reward_list)), key=lambda k: reward_list[k])

        if idxs[0] == 1 and dp_loss:
            supervise_cnt[0] += 1
            loss += dp_loss
        elif idxs[0] == 2 and tt_loss:
            supervise_cnt[1] += 1
            loss += tt_loss
        else:
            loss += (reward_list[0]-0.5*reward_list[1]-0.5*reward_list[2]) * rl_loss

        transition_rl_loss += rl_loss.item()

        if dp_loss:
            transition_cross_entropy_loss += dp_loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("--Training--\nEpoch: ", epoch, "Profiler Loss: ", profiler_loss, "Transition sentence loss: ",
          transition_cross_entropy_loss, "Transition RL", transition_rl_loss, "Time Elapsed: ", time.time()-start_time,
          "Supervised (IP, TT) ", supervise_cnt)

    perf = evaluate(validate_data)
    if prev_best_macro < perf:
        prev_best_macro = perf
        print ("-------------------Test start-----------------------")
        _ = evaluate(test_data, True)
        torch.save(model.state_dict(), '../final_model_1_{0}.pt'.format(seed))
        print ("-------------------Test end-----------------------")


def evaluate(data, is_test=False):
    y_true, y_pred = [], []
    trans_true, trans_pred = [], []
    model.eval()
    xxxx = {}

    for doc in data:
        sent, ls, out, sids, trans_sents, trans_sents_tt = get_batch(doc)
        if has_cuda:
            ls = ls.cuda()

        with torch.no_grad():
            _output, _, _, _, _, _, trans_scores = model.forward(sent, ls, is_test=True)
        _output = _output.squeeze()
        _output = _output[:-1]
        _, predict = torch.max(_output, 1)
        y_pred += list(predict.cpu().numpy() if has_cuda else predict.numpy())
        temp_true = list(out.numpy())
        y_true += temp_true

        trans_true += trans_sents[1:-1]
        xxxx[doc.name] = '__'.join([str(e-1) for e in trans_scores]) +'\t' + \
                         '__'.join([str(id) for id,e in enumerate(trans_sents) if e==1]) +'\t' + \
                         '__'.join([str(id) for id,e in enumerate(trans_sents_tt) if e==1])
        temp_pred = [0]*len(trans_sents)
        for elem in trans_scores:
            temp_pred[elem-1] = 1
        trans_pred += temp_pred[1:-1]

    print("MACRO: ", precision_recall_fscore_support(y_true, y_pred, average='macro'))
    print("MICRO: ", precision_recall_fscore_support(y_true, y_pred, average='micro'))
    print("Classification Report \n", classification_report(trans_true, trans_pred))
    if is_test:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Classification Report \n", classification_report(y_true, y_pred))

        print("Confusion Matrix \n", confusion_matrix(y_true, y_pred))
        print("Total Trans: ", sum(trans_pred))
    if not is_test:
        ftrans = open("trans_sents.txt", 'w')
        for key in xxxx:
            ftrans.write(key+'\t'+xxxx[key]+'\n')
    return precision_recall_fscore_support(y_true, y_pred, average='macro')[2]

if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--drop', help='DROP', default=6, type=float)
    parser.add_argument('--learn_rate', help='LEARNING RATE', default=0, type=float)
    parser.add_argument('--seed', help='SEED', default=0, type=int)

    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()

    SPEECH = 0
    if SPEECH:
        out_map = {'NA':0, 'Speech':1}
    else:
        out_map = {'NA':0,'Main':1,'Main_Consequence':2, 'Cause_Specific':3, 'Cause_General':4, 'Distant_Historical':5,
        'Distant_Anecdotal':6, 'Distant_Evaluation':7, 'Distant_Expectations_Consequences':8}

    train_data = []
    validate_data = []
    test_data = []
    for domain in ["Business", "Politics", "Crime", "Disaster", "kbp"]:
        subdir = "../data/train/"+domain
        files = os.listdir(subdir)
        for file in files:
            if '.txt' in file:
                doc = process_doc(os.path.join(subdir, file), domain) #'../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
                #print(doc.sent_to_event)
                train_data.append(doc)
        subdir = "../data/test/"+domain
        files = os.listdir(subdir)
        for file in files:
            if '.txt' in file:
                doc = process_doc(os.path.join(subdir, file), domain) #'../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
                #print(doc.sent_to_event)
                test_data.append(doc)


    subdir = "../data/validation"
    files = os.listdir(subdir)
    for file in files:
        if '.txt' in file:
            doc = process_doc(os.path.join(subdir, file), 'VAL') #'../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
            #print(doc.sent_to_event)
            validate_data.append(doc)
    print(len(train_data), len(validate_data), len(test_data))

    drop = args.drop
    learn_rate = args.learn_rate

    seed = args.seed
    print ("-------   drop ", drop, "learning rate ", learn_rate, "seed ", seed, "--------")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if has_cuda:
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    prev_best_macro = 0.

    model = Classifier({'num_layers': 1, 'hidden_dim': 512, 'bidirectional': True, 'embedding_dim': 1024,
                        'dropout': drop, 'out_dim': len(out_map)})

    if has_cuda:
        model = model.cuda()
    model.init_weights()

    criterion = nn.CrossEntropyLoss()

    print("Model Created")

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=learn_rate, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)

    try:
        for epoch in range(15):
            print("---------------------------Started Training Epoch = {0}--------------------------".format(epoch+1))
            train(epoch, train_data, seed)
        torch.save(model.state_dict(), 'RL_IP_TT.pt')

    except KeyboardInterrupt:
        print ("----------------- INTERRUPTED -----------------")
        evaluate(validate_data)
        evaluate(test_data)
from collections import OrderedDict
from nltk.tokenize import TextTilingTokenizer


ttt = TextTilingTokenizer()


def sort_order(sids):
    temp = sorted([int(elem[1:]) for elem in sids])
    return ['S'+str(elem) for elem in temp]


class Document(object):

    def __init__(self, fname, domain):
        self.name = fname.split('/')[-1]
        self.domain = domain
        self.url = 'None'
        self.headline = ['headline']
        self.lead = []
        self.sentences = OrderedDict()
        self.tags = dict()
        self.sent_to_speech = None
        self.sent_to_event = None
        self.sent_to_topic = None
        self.sent_to_topic_text_tile = None
        self.sids = []


def process_doc(fname, domain):
    out_map = {'NA': 0, 'Main': 1, 'Main_Consequence': 2, 'Cause_Specific': 2, 'Cause_General': 2,
               'Distant_Historical': 3, 'Distant_Anecdotal': 3, 'Distant_Evaluation': 3, 'Distant_Expectations_Consequences': 3}
    # Process text document
    f = open(fname, 'r')
    # print(fname)
    doc = Document(fname, domain)
    lead_para = False
    sids = []
    text = ''
    for line in f:
        temp = line.strip()
        if temp == '':
            text += '\n'
            if lead_para == True:
                for key in doc.sentences:
                    doc.lead += doc.sentences[key]
            lead_para = False
            continue

        temp = temp.split()
        if temp[0] == 'URL':
            doc.url = temp[1]
        elif temp[0] == 'DATE/':
            pass
        elif temp[0] == 'H':
            text += ' '.join(temp[1:]) + '\n'
            if len(temp[1:]) > 0:
                doc.headline = temp[1:]
        else:
            if temp[0] == 'S1':
                lead_para = True
            doc.sentences[temp[0]] = temp[1:]
            sids.append(temp[0])
            text += ' '.join(temp[1:]) + '\n'

    # Process annotation file
    f = open(fname[:-3]+'ann')
    # prev_label = "headline"
    sent_to_event = dict()
    sent_to_speech = dict()
    sent_to_topic = dict()
    sent_to_topic_tt = dict()
    for line in f:
        temp = line.strip().split('\t')
        if len(temp) == 3:
            label = temp[1].split()[0]
            if label == 'Speech':
                sent_to_speech[temp[2]] = label
            else:
                # print(temp)
                sent_to_event[temp[2]] = label
                sent_to_topic[temp[2]] = 0
                sent_to_topic_tt[temp[2]] = 0

    doc.sent_to_event = sent_to_event
    doc.sent_to_speech = sent_to_speech
    doc.sids = sort_order(sids)
    prev_label = 'NA'
    for ind in range(len(doc.sids)):
        if prev_label not in ['Main', 'Main_Consequence'] and 'Main' in sent_to_event[doc.sids[ind]]:  # 'Distant_Anecdotal',
            sent_to_topic[doc.sids[ind]] = 1
        elif prev_label not in ['Main', 'Main_Consequence', 'Cause_General', 'Cause_Specific'] and 'Cause' in sent_to_event[doc.sids[ind]]:
            sent_to_topic[doc.sids[ind]] = 1
        elif prev_label != sent_to_event[doc.sids[ind]] and sent_to_event[doc.sids[ind]] == 'Distant_Anecdotal':
            sent_to_topic[doc.sids[ind]] = 1
        elif prev_label != sent_to_event[doc.sids[ind]] and prev_label == 'Distant_Anecdotal' and sent_to_event[doc.sids[ind]] != 'NA':
            sent_to_topic[doc.sids[ind]] = 1
        elif prev_label != sent_to_event[doc.sids[ind]] and sent_to_event[doc.sids[ind]] == 'Distant_Historical':
            sent_to_topic[doc.sids[ind]] = 1
        elif prev_label != sent_to_event[doc.sids[ind]] and prev_label == 'Distant_Historical' and sent_to_event[doc.sids[ind]] != 'NA':
            sent_to_topic[doc.sids[ind]] = 1

        if sent_to_event[doc.sids[ind]] != 'NA':
            prev_label = sent_to_event[doc.sids[ind]]

    assert(len(sent_to_event) == len(doc.sids))
    doc.sent_to_topic = sent_to_topic

    try:
        tokens = ttt.tokenize(text)
        all_trans = []
        for token in tokens:
            x = token.split('\n')
            label = 1
            for s in x:
                if s != '':
                    all_trans.append((label, s))
                    label = 0

        if len(all_trans) == len(sids):
            all_trans = [(0,'')]+all_trans
        assert (len(all_trans) == 1+len(sids))
        for sind,sid in enumerate(doc.sids):
            if sind == 0:
                sent_to_topic_tt[sid] = 1
            elif all_trans[sind+1][0] == 1:
                if ' '.join(doc.sentences[sid]) == all_trans[sind+1][1]:
                    sent_to_topic_tt[sid] = 1
                else:
                    print(' '.join(doc.sentences[sid]))
                    print("-", all_trans[sind][1])
    except:
        print(text)
    doc.sent_to_topic_text_tile = sent_to_topic_tt
    # print(sent_to_topic)
    return doc

# process_doc('../data/validation/ACXinhua_16.txt', 'valid')


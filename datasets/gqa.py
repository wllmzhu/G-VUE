import torch
import torch.utils.data as data
import json
import numpy as np
import os
import pickle
from PIL import Image

class Dictionary(object):

    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx, self.idx2word = word2idx, idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower().replace(',', '').replace('.', '').replace('?', '').replace('\'s', ' \'s')
        words, tokens = sentence.split(), []

        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx.get(w, self.padding_idx))

        return tokens

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def gqa_create_dictionary_glove(gqa_q='data/GQA-Questions', glove='data/GloVe/glove.6B.300d.txt',
                                cache='data/GQA-Cache'):

    dfile, gfile = os.path.join(cache, 'dictionary.pkl'), os.path.join(cache, 'glove.npy')
    if os.path.exists(dfile) and os.path.exists(gfile):
        with open(dfile, 'rb') as f:
            dictionary = pickle.load(f)

        weights = np.load(gfile)
        return dictionary, weights

    elif not os.path.exists(cache):
        os.makedirs(cache)

    dictionary = Dictionary()
    questions = ['train_balanced_questions.json', 'val_balanced_questions.json', 'testdev_balanced_questions.json',
                 'test_balanced_questions.json']

    print('\t[*] Creating Dictionary from GQA Questions...')
    for qfile in questions:
        qpath = os.path.join(gqa_q, qfile)
        with open(qpath, 'r') as f:
            examples = json.load(f)

        for ex_key in examples:
            ex = examples[ex_key]
            dictionary.tokenize(ex['question'], add_word=True)

    print('\t[*] Loading GloVe Embeddings...')
    with open(glove, 'r') as f:
        entries = f.readlines()

    assert len(entries[0].split()) - 1 == 300, 'ERROR - Not using 300-dimensional GloVe Embeddings!'

    weights = np.zeros((len(dictionary.idx2word), 300), dtype=np.float32)

    for entry in entries:
        word_vec = entry.split()
        word, vec = word_vec[0], list(map(float, word_vec[1:]))
        if word in dictionary.word2idx:
            weights[dictionary.word2idx[word]] = vec

    with open(dfile, 'wb') as f:
        pickle.dump(dictionary, f)
    np.save(gfile, weights)

    return dictionary, weights


def gqa_create_answers(gqa_q='data/GQA-Questions', cache='data/GQA-Cache'):

    dfile = os.path.join(cache, 'answers.pkl')
    if os.path.exists(dfile):
        with open(dfile, 'rb') as f:
            ans2label, label2ans = pickle.load(f)

        return ans2label, label2ans

    ans2label, label2ans = {}, []
    questions = ['train_balanced_questions.json', 'val_balanced_questions.json', 'testdev_balanced_questions.json']

    print('\t[*] Creating Answer Labels from GQA Question/Answers...')
    for qfile in questions:
        qpath = os.path.join(gqa_q, qfile)
        with open(qpath, 'r') as f:
            examples = json.load(f)

        for ex_key in examples:
            ex = examples[ex_key]
            if not ex['answer'].lower() in ans2label:
                ans2label[ex['answer'].lower()] = len(ans2label)
                label2ans.append(ex['answer'])

    with open(dfile, 'wb') as f:
        pickle.dump((ans2label, label2ans), f)

    return ans2label, label2ans


class GQADataset(data.Dataset):
    """Dataloader for GQA Dataset"""

    def __init__(self, dictionary, ans2label, label2ans, gqa_q='data/GQA-Questions', img_dir='data/GQA-Images',
                 mode='train'):
        super(GQADataset, self).__init__()
        self.dictionary, self.ans2label, self.label2ans = dictionary, ans2label, label2ans

        self.entries = load_dataset(ans2label, gqa_q=gqa_q, img_dir=img_dir, mode=mode)

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=40):
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            assert len(tokens) == max_length, "Tokenized & Padded Question != Max Length!"
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        target = entry['answer']
        image = entry['image']

        return image, question, target

    def __len__(self):
        return len(self.entries)



def load_dataset(ans2label, gqa_q='data/GQA-Questions', img_dir='data/GQA-Images', mode='train'):

    question_path = os.path.join(gqa_q, '%s_balanced_questions.json' % mode)
    with open(question_path, 'r') as f:
        examples = json.load(f)

    print('\t[*] Creating GQA %s Entries...' % mode)
    entries = []
    for ex_key in sorted(examples):
        entry = create_entry(examples[ex_key], ex_key, ans2label, img_dir)
        entries.append(entry)

    return entries

def create_entry(example, qid, ans2label, img_dir):
    img_id = example['imageId']

    entry = {
        'question_id': qid,
        'image_id': img_id,
        'image': Image.open(os.path.join(img_dir, img_id)).convert("RGB"),
        'question': example['question'],
        'answer': ans2label[example['answer'].lower()]
    }
    return entry


def main():
    print('\n[*] Pre-processing GQA Questions...')
    dictionary, emb = gqa_create_dictionary_glove(gqa_q='data/GQA-Questions', glove='data/GloVe/glove.6B.300d.txt', cache='data/GQA-Cache')

    print('\n[*] Pre-processing GQA Answers...')
    ans2label, label2ans = gqa_create_answers(gqa_q='data/GQA-Questions', cache='data/GQA-Cache')

    print('\n[*] Building GQA Train and TestDev Datasets...')
    train_dataset = GQADataset(dictionary, ans2label, label2ans, gqa_q='data/GQA-Questions', mode='train')

    dev_dataset = GQADataset(dictionary, ans2label, label2ans, gqa_q='data/GQA-Questions', mode='testdev')

if __name__ == '__main__':
    main()
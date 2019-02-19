import tensorflow as tf
import numpy as np
import pickle
import json
import re

from config import *
from sklearn.utils import shuffle

class Vocab:
    def __init__(self, token2idx=None, idx2token=None):
        self.token2idx = token2idx or dict()
        self.idx2token = idx2token or dict()

    def new_token(self, token):
        # token : word or character
        # return : index of token
        if token not in self.token2idx:
            index = len(self.token2idx)
            self.token2idx[token] = index
            self.idx2token[index] = token
        return self.token2idx[token]

    def get_token(self, index):
        # index : position number of token
        # return : word or character of index
        return self.idx2token[index]

    def get_index(self, token):
        # token : word or character
        # return : index of token
        return self.token2idx[token]

def clean_str(text):
    text = re.sub('<br>', '', text)
    text = re.sub('</b>', '', text)
    text = re.sub(':', ' :', text)
    text = re.sub(',', ' ,', text)
    text = re.sub('"', ' " ', text)
    text = re.sub(r'\.', ' .', text)
    text = re.sub(r'\?', ' ?', text)
    text = re.sub(r'\-', ' - ', text)
    text = re.sub(r'\(', '( ', text)
    text = re.sub(r'\)', ' )', text)
    text = re.sub(r'\'', ' \' ', text)
    text = re.sub(r'\;', ' ;', text)
    text = re.sub('n\'t', ' n\'t ', text)
    text = re.sub(r'[', '[ ', text)
    text = re.sub(r']', ' ]', text)
    text = re.sub(r'!', ' !', text)
    return text

def make_data():
    with open(FLAGS.data_dir + FLAGS.train_file + '.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    preprocess_data = dict()
    for idx in range(len(data['data'])):
        preprocess_data[idx] = dict()

        preprocess_data[idx]['id'] = data['data'][idx]['id']
        questions = data['data'][idx]['paragraph']['questions']
        text = data['data'][idx]['paragraph']['text']
        text = clean_str(text)

        # all sentences in list
        all_sentences = text.split('<b>')[1:]
        preprocess_data[idx]['paragraph'] = all_sentences

        preprocess_data[idx]['questions'] = dict()
        for ques_idx in range(len(questions)):
            preprocess_data[idx]['questions'][ques_idx] = dict()
            preprocess_data[idx]['questions'][ques_idx]['query'] = questions[ques_idx]['question']
            preprocess_data[idx]['questions'][ques_idx]['sentence_used'] = questions[ques_idx]['sentences_used']
            preprocess_data[idx]['questions'][ques_idx]['answers'] = questions[ques_idx]['answers']
            preprocess_data[idx]['questions'][ques_idx]['multi_sentence'] = questions[ques_idx]['multisent']

    return preprocess_data

# load pretrained GLOVE file
def make_glove():
    word_vocab = dict()
    with open(FLAGS.glove_dir + FLAGS.glove_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            one_line = line.split()
            try:
                word = one_line[0]
                embedding = np.array([float(val) for val in one_line[1:]])
                word_vocab[word] = embedding

            except ValueError as e:
                if idx == 52343:
                    word = '. . .'
                    embedding = np.array([float(val) for val in one_line[3:]])
                    word_vocab[word] = embedding
                elif idx == 151102:
                    word = '. . . . .'
                    embedding = np.array([float(val) for val in one_line[5:]])
                    word_vocab[word] = embedding
                elif idx == 209833:
                    word = '. .'
                    embedding = np.array([float(val) for val in one_line[2:]])
                    word_vocab[word] = embedding
                elif idx == 220779:
                    word = '. . . .'
                    embedding = np.array([float(val) for val in one_line[4:]])
                    word_vocab[word] = embedding

                if idx in FLAGS.except2_idx:
                    one_line = line.split()
                    word = one_line[0] + ' ' + one_line[1]
                    embedding = np.array([float(val) for val in one_line[2:]])
                    word_vocab[word] = embedding

        print ('Done Loading Glove model')
    return word_vocab

def pickle_dump(glove_vocab=None):
    if glove_vocab is None:
        print ('Loading pickle to dictionary')
        with open(FLAGS.glove_dir + FLAGS.glove_load, 'rb') as f:
            glove_vocab = pickle.load(f)
        return glove_vocab
    else:
        print ('Start Writing dictionary to pickle')
        with open(FLAGS.glove_dir + FLAGS.glove_load, 'wb') as f:
            pickle.dump(glove_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

def vocab_data(preprocess_data):
    # --------------------------- Input --------------------------- #
    # preprocess_data : preprocessed data, which has paragraph, answer, query

    # --------------------------- Output --------------------------- #
    # sentence_word : it contains paragraph words per index
    # sentence_char : it contains paragraph characters per index
    # query_word : it contain query per paragraph related to words info
    # query_char : it contains query per paragraph related to characters info
    # word_vocab : token2idx and idx2token word dictionary
    # char_vocab : token2idx and idx2token character dictionary
    # word_maxlen : max character length (which is word length)

    word_vocab = Vocab()
    char_vocab = Vocab()

    word_vocab.new_token('UNK')

#     EOS = '|'
#     word_vocab.new_token(EOS)
#     char_vocab.new_token(EOS)

    sentence_word = dict()
    sentence_char = dict()
    query_word = dict()
    query_char = dict()

    # word max length
    word_maxlen = 0

    for idx in range(len(preprocess_data)):
        paragraph = preprocess_data[idx]['paragraph']
        sentence_word[idx] = list()
        sentence_char[idx] = list()

        queries = preprocess_data[idx]['questions']
        query_word[idx] = list()
        query_char[idx] = list()

        # paragraph preprocessing
        for sent_idx, sentence in enumerate(paragraph):

            for word in sentence.split():
                # word token into dictionary
                sentence_word[idx].append(word_vocab.new_token(word))

                # character token into dictionary
                sentence_char[idx].append([char_vocab.new_token(c) for c in word])

                if len(word) > word_maxlen:
                    word_maxlen = len(word)

#             sentence_word[idx].append(word_vocab.get_index(EOS))
#             sentence_char[idx].append(char_vocab.get_index(EOS))

        # query preprocessing
        for query_idx in queries:
            query_word_list = list()
            query_char_list = list()
            query_dict = queries[query_idx]
            query_parsed = clean_str(query_dict['query'])

            for word in query_parsed.split():
                # word token into dictionary
                query_word_list.append(word_vocab.new_token(word))
                #query_word[idx].append(word_vocab.new_token(word))

                # character token into dictionary
                query_char_list.append([char_vocab.new_token(c) for c in word])
                #query_char[idx].append([char_vocab.new_token(c) for c in word])

                if len(word) > word_maxlen:
                    word_maxlen = len(word)

            query_word[idx].append(query_word_list)
            query_char[idx].append(query_char_list)

    return sentence_word, sentence_char, query_word, query_char, word_vocab, char_vocab, word_maxlen

def glove_idx2vec(word_vocab, glove_vocab):
    word_idx2vec = dict()

    for word, idx in word_vocab.token2idx.items():
        try:
            word_idx2vec[word] = glove_vocab[word]
        except KeyError as e:
            word_idx2vec[word] = np.zeros((300), dtype=np.float64)
    return word_idx2vec

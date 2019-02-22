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
    text = re.sub(r'\[', '[ ', text)
    text = re.sub(r'\]', ' ]', text)
    text = re.sub(r'\!', ' !', text)
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
        print ('Loading Glove pickle to dictionary')
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
    answer_text = dict()
    answer_text_char = dict()

    # word max length
    word_maxlen = 0
    answer_maxnum = 0

    for idx in range(len(preprocess_data)):
        paragraph = preprocess_data[idx]['paragraph']
        sentence_word[idx] = list()
        sentence_char[idx] = list()

        queries = preprocess_data[idx]['questions']
        query_word[idx] = list()
        query_char[idx] = list()
        answer_text[idx] = dict()
        answer_text_char[idx] = dict()

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
            answer_text[idx][query_idx] = dict()
            answer_text_char[idx][query_idx] = dict()

            query_dict = queries[query_idx]
            query_parsed = clean_str(query_dict['query'])
            answer_list = queries[query_idx]['answers']

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

            # Answer preprocessing
            for answer_idx, answer in enumerate(answer_list):
                answer_text[idx][query_idx][answer_idx] = list()
                answer_text_char[idx][query_idx][answer_idx] = list()
                answer_parsed = clean_str(answer['text'])

                for word in answer_parsed.split():
                    # word token into dictionary
                    answer_text[idx][query_idx][answer_idx].append(word_vocab.new_token(word))

                    # character token into dictionary
                    answer_text_char[idx][query_idx][answer_idx].append([char_vocab.new_token(c) for c in word])

                    if len(word) > word_maxlen:
                        word_maxlen = len(word)

            if len(answer_list) + 1 > answer_maxnum:
                answer_maxnum = len(answer_list) + 1

    return sentence_word, sentence_char, query_word, query_char, answer_text, answer_text_char, \
            word_vocab, char_vocab, word_maxlen, answer_maxnum

def make_answer_dict(preprocess_data, answer_text, answer_text_char):

    answer_text_presence = dict()

    for idx in preprocess_data:
        answer_text_presence[idx] = dict()
        queries = preprocess_data[idx]['questions']

        for query_idx in queries:
            answer_text_presence[idx][query_idx] = dict()
            # answer list with (text, isAnswer, scores)
            answers = queries[query_idx]['answers']

            for answer_idx, answer in enumerate(answers):
                answer_text_presence[idx][query_idx][answer_idx] = dict()
                answer_text_presence[idx][query_idx][answer_idx]['isAnswer'] = 1.0 if answer['isAnswer'] else 0.0
                answer_text_presence[idx][query_idx][answer_idx]['text'] = answer_text[idx][query_idx][answer_idx]
                answer_text_presence[idx][query_idx][answer_idx]['char_text'] = answer_text_char[idx][query_idx][answer_idx]

    return answer_text_presence

def glove_idx2vec(word_vocab, glove_vocab):
    word_idx2vec = dict()

    for word, idx in word_vocab.token2idx.items():
        try:
            word_idx2vec[word] = glove_vocab[word]
        except KeyError as e:
            word_idx2vec[word] = np.zeros((300), dtype=np.float64)
    return word_idx2vec

def embedding_matrix(sentence_word, sentence_char, answer_text_presence, query_word, query_char, word_maxlen):
    sentence_maxlen = 92

    word_matrix = dict()
    query_matrix = dict()
    char_matrix = dict()
    answer_matrix = dict()
    char_answer_matrix = dict()

    query_matrix['word'] = dict()
    query_matrix['char'] = dict()

    for sent_idx in sentence_word:
        # paragraph word token
        word_matrix[sent_idx] = np.array(sentence_word[sent_idx], dtype=np.float32)

        # query word token
        query_matrix['word'][sent_idx] = dict()
        for query_idx, query in enumerate(query_word[sent_idx]):
            query_matrix['word'][sent_idx][query_idx] = np.array(query_word[sent_idx][query_idx], dtype=np.float32)

        # paragraph character token
        char_matrix[sent_idx] = np.zeros([len(sentence_char[sent_idx]), word_maxlen], dtype=np.float32)
        for word_idx, char_list in enumerate(sentence_char[sent_idx]):
            char_matrix[sent_idx][word_idx, :len(char_list)] = char_list

        # query character token
        query_matrix['char'][sent_idx] = dict()
        for query_idx, query in enumerate(query_char[sent_idx]):
            query_matrix['char'][sent_idx][query_idx] = np.zeros([len(query_char[sent_idx][query_idx]), word_maxlen], dtype=np.float32)
            for word_idx, char_list in enumerate(query_char[sent_idx][query_idx]):
                query_matrix['char'][sent_idx][query_idx][word_idx, :len(char_list)] = char_list

        # answer word token
        answer_matrix[sent_idx] = dict()
        for query_idx in answer_text_presence[sent_idx]:
            # answer text max length = 92
            answer_matrix[sent_idx][query_idx] = dict()
            answer_matrix[sent_idx][query_idx]['isAnswer'] = np.zeros([len(answer_text_presence[sent_idx][query_idx]), 2], \
                                                                     dtype=np.float32)
            answer_matrix[sent_idx][query_idx]['text'] = np.zeros([len(answer_text_presence[sent_idx][query_idx]), sentence_maxlen], \
                                                                  dtype=np.int32)

            for query_by_answer_idx in answer_text_presence[sent_idx][query_idx]:
                query_ans_text = answer_text_presence[sent_idx][query_idx][query_by_answer_idx]['text']
                answer_matrix[sent_idx][query_idx]['text'][query_by_answer_idx, :len(query_ans_text)] = query_ans_text

                query_ans_isAnswer = answer_text_presence[sent_idx][query_idx][query_by_answer_idx]['isAnswer']
                if query_ans_isAnswer:
                    answer_matrix[sent_idx][query_idx]['isAnswer'][query_by_answer_idx] = [1.0, 0.0]
                else:
                    answer_matrix[sent_idx][query_idx]['isAnswer'][query_by_answer_idx] = [0.0, 1.0]


        # answer character token
        char_answer_matrix[sent_idx] = dict()
        for query_idx in answer_text_presence[sent_idx]:
            # answer text max length = 92
            char_answer_matrix[sent_idx][query_idx] = dict()

            for query_by_answer_idx in answer_text_presence[sent_idx][query_idx]:
                text_num = answer_text_presence[sent_idx][query_idx][query_by_answer_idx]
                char_answer_matrix[sent_idx][query_idx][query_by_answer_idx] = \
                np.zeros([len(text_num['text']), word_maxlen], dtype=np.float32)

                for char_text_idx in range(len(text_num['text'])):
                    char_answer_matrix[sent_idx][query_idx][query_by_answer_idx][char_text_idx, :len(text_num['char_text'][char_text_idx])] = \
                    text_num['char_text'][char_text_idx]

    print ('Shape of Embedding Matrix')
    print ('One Word Matrix shape : ', word_matrix[0].shape)
    print ('One Character Matrix shape : ', char_matrix[0].shape)

    print ('One Query Word Matrix shape :', query_matrix['word'][0][0].shape)
    print ('One Query Character Matrix shape : ', query_matrix['char'][0][0].shape)

    print ('Answer Text Matrix shape : ', answer_matrix[0][0]['text'][0].shape)
    print ('Answer presence Matrix shape : ', answer_matrix[0][0]['isAnswer'][0].shape)

    print ('Answer Text Character Matrix shape : ', char_answer_matrix[0][0][0].shape)

    return word_matrix, char_matrix, query_matrix, answer_matrix, char_answer_matrix

def preproecessing():
    glove_vocab = pickle_dump(glove_vocab=None)
    preprocess_data = make_data()

    sentence_word, sentence_char, query_word, query_char, answer_text, answer_text_char, word_vocab, \
    char_vocab, word_maxlen, answer_maxnum = vocab_data(preprocess_data)

    answer_text_presence = make_answer_dict(preprocess_data, answer_text, answer_text_char)
    word_idx2vec = glove_idx2vec(word_vocab, glove_vocab)

    paragraph_matrix, char_matrix, query_matrix, answer_matrix, char_answer_matrix = \
    embedding_matrix(sentence_word, sentence_char, answer_text_presence, query_word, query_char, word_maxlen)

    # batch loader

    # zip input file - one paragraph, one query, one answer
    return paragraph_matrix, char_matrix, query_matrix, answer_matrix, char_answer_matrix

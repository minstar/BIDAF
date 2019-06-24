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
    text = re.sub('<b>', ' ', text)
    text = re.sub(':', ' :', text)
    text = re.sub(',', ' ,', text)
    text = re.sub('"', ' " ', text)
    text = re.sub(r'\.', ' .', text)
    text = re.sub(r'\?', ' ?', text)
    text = re.sub(r'\-', ' - ', text)
    text = re.sub(r'\(', '( ', text)
    text = re.sub(r'\)', ' )', text)
    text = re.sub(r'\;', ' ;', text)
    text = re.sub('n\'t', ' not ', text) # heuristic for glove word table
    text = re.sub(r'\[', '[ ', text)
    text = re.sub(r'\]', ' ]', text)
    text = re.sub(r'\!', ' !', text)
    return text

# load pretrained GLOVE file
def make_glove():
    word_vocab = dict()
    start = time.time()

    with open(FLAGS.glove_dir + FLAGS.glove_file, 'r', encoding='utf-8') as reader:
        print ('Start Making Glove word dictionary')
        for idx, line in enumerate(reader):
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

            if (idx+1) % 1000000 == 0:
                print ('elapsed time for making 100000 word', time.time() - start)

        print ('Done Loading Glove word dictionary')
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


def multirc_data_load():

    with open('./dataset/splitv2/train_456-fixedIds.json', 'r', encoding='utf-8') as reader:
        train_data = json.load(reader)

    with open('./dataset/splitv2/dev_83-fixedIds.json', 'r', encoding='utf-8') as reader:
        dev_data = json.load(reader)

    return train_data, dev_data

def make_data(train_data):

    whole_word = dict()
    whole_char = dict()
    query_word = dict()
    query_char = dict()
    label_dict = dict()
    label_word = dict()

    word_maxlen = 0
    ans_maxnum = 0

    char_vocab = Vocab()
    word_vocab = Vocab()

    word_vocab.new_token('UNK')

    for data_idx in range(len(train_data['data'])):
        whole_word[data_idx] = list()
        whole_char[data_idx] = list()

        query_word[data_idx] = dict()
        query_char[data_idx] = dict()

        label_dict[data_idx] = dict()
        label_word[data_idx] = dict()

        # one text and many questions
        # In one question, it has many answers
        text = clean_str(train_data['data'][data_idx]['paragraph']['text'])
        text = text.split('Sent')[1:]
        questions = train_data['data'][data_idx]['paragraph']['questions']

        # text : make word vocabulary dictionary and character vocabulary dictionary
        for sent_idx in range(len(text)):
            new_sent = text[sent_idx][5:].split()

            for word in new_sent:
                whole_word[data_idx].append(word_vocab.new_token(word))

                whole_char[data_idx].append([char_vocab.new_token(c) for c in word])

                if len(word) > word_maxlen:
                    word_maxlen = len(word)

        # question : make question word vocabulary dictionary and character vocabulary dictionary
        for ques_idx in range(len(questions)):
            new_ques = clean_str(questions[ques_idx]['question'])
            label_dict[data_idx][ques_idx] = dict()
            label_word[data_idx][ques_idx] = dict()

            query_word[data_idx][ques_idx] = list()
            query_char[data_idx][ques_idx] = list()

            for word in new_ques.split():
                query_word[data_idx][ques_idx].append(word_vocab.new_token(word))

                query_char[data_idx][ques_idx].append([char_vocab.new_token(c) for c in word])

                if len(word) > word_maxlen:
                    word_maxlen = len(word)

            # answer text and answer presence
            new_ans = questions[ques_idx]['answers']

            for ans_idx in range(len(new_ans)):
                label_word[data_idx][ques_idx][ans_idx] = list()
                label_dict[data_idx][ques_idx][ans_idx] = np.zeros((2), dtype=np.int32)

                ans_text = clean_str(new_ans[ans_idx]['text'])
                # answer presence
                if new_ans[ans_idx]['isAnswer']:
                    label_dict[data_idx][ques_idx][ans_idx] = [1, 0]
                else:
                    label_dict[data_idx][ques_idx][ans_idx] = [0, 1]

                # answer text if character need then you need to add the code
                for word in ans_text.split():
                    label_word[data_idx][ques_idx][ans_idx].append(word_vocab.new_token(word))

            if ans_maxnum < len(new_ans):
                ans_maxnum = len(new_ans)

    return word_vocab, char_vocab, whole_word, whole_char, query_word, query_char, label_dict, label_word, word_maxlen, ans_maxnum

def glove_idx2vec(word_vocab, glove_vocab):
    word_idx2vec = np.zeros((len(word_vocab.token2idx), 300), dtype=np.float32)

    for word, idx in word_vocab.token2idx.items():
        try:
            word_idx2vec[idx, :] = glove_vocab[word]
        except KeyError as e:
            word_idx2vec[idx, :] = glove_vocab['unk']

    return word_idx2vec

def embedding_matrix(whole_word, whole_char, query_word, query_char, label_word):
    text_word = np.zeros((456, FLAGS.max_text), dtype=np.int32)
    text_char = np.zeros((456, FLAGS.max_text, FLAGS.max_char), dtype=np.int32)
    question_word = np.zeros((456, FLAGS.max_num_query, FLAGS.max_query_text), dtype=np.int32)
    question_char = np.zeros((456, FLAGS.max_num_query, FLAGS.max_query_text, FLAGS.max_query_char), dtype=np.int32)
    answer_word = np.zeros((456, FLAGS.max_num_query, FLAGS.max_num_answer, FLAGS.max_answer), dtype=np.int32)

    for data_idx in range(len(whole_word)):
        # paragraph text word array
        text_word[data_idx, :len(whole_word[data_idx])] = whole_word[data_idx]

        # paragraph text character array
        for word_idx in range(len(whole_char[data_idx])):
            text_char[data_idx, word_idx, :len(whole_char[data_idx][word_idx])] = whole_char[data_idx][word_idx]

        # question text word array
        for ques_idx in range(len(query_word[data_idx])):
            question_word[data_idx, ques_idx, :len(query_word[data_idx][ques_idx])] = query_word[data_idx][ques_idx]

        # question text character array
        for ques_idx in range(len(query_char[data_idx])):
            for word_idx in range(len(query_char[data_idx][ques_idx])):
                question_char[data_idx, ques_idx, word_idx, :len(query_char[data_idx][ques_idx][word_idx])] = query_char[data_idx][ques_idx][word_idx]

        # answer text word array
        for ques_idx in range(len(label_word[data_idx])):
            for ans_idx in range(len(label_word[data_idx][ques_idx])):
                answer_word[data_idx, ques_idx, ans_idx, :len(label_word[data_idx][ques_idx][ans_idx])] = label_word[data_idx][ques_idx][ans_idx]

    print ('Text paragraph word array size', text_word.shape)
    print ('Text paragraph character array size', text_char.shape)
    print ('Question word array size', question_word.shape)
    print ('Question character array size', question_char.shape)
    print ('Answer word array size', answer_word.shape)

    return text_word, text_char, question_word, question_char, answer_word

def batch_loader(text_word, text_char, question_word, question_char, answer_word):

    reduced_length = (len(text_word) // FLAGS.batch_size) * FLAGS.batch_size

    text_word = text_word[:reduced_length]
    text_char = text_char[:reduced_length]
    question_word = question_word[:reduced_length]
    question_char = question_char[:reduced_length]
    answer_word = answer_word[:reduced_length]

    text_word = np.reshape(text_word, [-1, FLAGS.batch_size, FLAGS.max_text])
    text_char = np.reshape(text_char, [-1, FLAGS.batch_size, FLAGS.max_text, FLAGS.max_char])
    question_word = np.reshape(question_word, [-1, FLAGS.batch_size, FLAGS.max_num_query, FLAGS.max_query_text])
    question_char = np.reshape(question_char, [-1, FLAGS.batch_size, FLAGS.max_num_query, FLAGS.max_query_text, FLAGS.max_query_char])
    answer_word = np.reshape(answer_word, [-1, FLAGS.batch_size, FLAGS.max_num_query, FLAGS.max_num_answer, FLAGS.max_answer])

    return text_word, text_char, question_word, question_char, answer_word

def random_shuffle(text_word, text_char, question_word, question_char, answer_word):

    text_word, text_char, question_word, question_char, answer_word = \
    shuffle(text_word, text_char, question_word, question_char, answer_word)

    return text_word, text_char, question_word, question_char, answer_word

def zip_file(text_word, text_char, question_word, question_char, answer_word):
    text_word = list(text_word)
    text_char = list(text_char)
    question_word = list(question_word)
    question_char = list(question_char)
    answer_word = list(answer_word)

    zip_list = list(zip(text_word, text_char, question_word, question_char, answer_word))

    return zip_list

def load_data():
    # make glove pickle file
    # glove_vocab = make_glove()
    glove_vocab = pickle_dump(glove_vocab=None)
    # data load
    train_data, dev_data = multirc_data_load()
    # make data from original data
    word_vocab, char_vocab, whole_word, whole_char, query_word, query_char,\
    label_dict, label_word, word_maxlen, ans_maxnum = make_data(train_data)
    # glove word2vec table
    word_idx2vec = glove_idx2vec(word_vocab, glove_vocab)
    # make list to array
    text_word, text_char, question_word, question_char, answer_word = embedding_matrix(whole_word, whole_char, query_word, query_char, label_word)
    # make as batch file
    text_word, text_char, question_word, question_char, answer_word = batch_loader(text_word, text_char, question_word, question_char, answer_word)
    # random shuffling
    text_word, text_char, question_word, question_char, answer_word = random_shuffle(text_word, text_char, question_word, question_char, answer_word)
    # zip all file to one sample
    zip_list = zip_file(text_word, text_char, question_word, question_char, answer_word)

    return zip_list, word_idx2vec

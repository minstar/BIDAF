import os
import re
import pdb
import json
import nltk
import pickle
import pprint
import numpy as np

from nltk.tokenize import word_tokenize
# for sanity check
from config import get_args
from utils import _seq2seq, process_tokens

class Squad_Dataset():
    def __init__(self, config):
        self.config = config
        self.word2idx = dict()
        self.idx2word = dict()
        self.char2idx = dict()
        self.idx2char = dict()
        self.train_data = dict()
        self.dev_data = dict()

        if self.config.is_load == 'True':
            self._load()
            self._read(file_name=self.train_file)
            self.train_data = self._max_num_find(file_name=self.train_data)
            self.config.mode = 'dev'
            self._read(file_name=self.dev_file)
            self.dev_data = self._max_num_find(file_name=self.dev_data)

        else:
            self._load()
            self._read(file_name=self.train_file)
            self.train_data = self._max_num_find(file_name=self.train_data)
            self._make_numpy(data_name=self.train_data)

            self.config.mode = 'dev'
            self._read(file_name=self.dev_file)
            self.dev_data = self._max_num_find(file_name=self.dev_data)
            self._make_numpy(data_name=self.dev_data)

    def _load(self, ):
        ### load glove and squad data
        with open(self.config.train_dir + self.config.train_file, 'r') as fp:
            self.train_file = json.load(fp)['data']
        with open(self.config.train_dir + self.config.dev_file, 'r') as fp:
            self.dev_file = json.load(fp)['data']
        with open(self.config.glove_dir + self.config.glove_dict, 'rb') as fp:
            self.glove_file = pickle.load(fp)

        if self.config.is_load == 'True':
            with open('tr_zip_list.pkl', 'rb') as fp:
                self.tr_zip_list = pickle.load(fp)
            with open('dev_zip_list.pkl', 'rb') as fp:
                self.dev_zip_list = pickle.load(fp)

    def _read(self, file_name=None):

        train_idx = 0
        dev_idx = 0
        print (" # --------------- start making data --------------- # \n")
        self.word2idx['pad'] = 0
        self.idx2word[0] = 'pad'
        self.word2idx['unk'] = 1
        self.idx2word[1] = 'unk'
        answer_mismatch = 0

        for idx in range(len(file_name)):
            paragraphs = file_name[idx]['paragraphs']

            for par_idx, paragraph in enumerate(paragraphs):
                context_list = list()
                context_char_list = list()

                context = paragraph['context'].replace("''", '" ').replace("``", '" ')
                qas = paragraph['qas']
                tokens = word_tokenize(context)
                # tokens = process_tokens(tokens)

                # context
                for token in tokens:
                    char_list = list()

                    if token not in self.word2idx:
                        self.word2idx[token] = len(self.word2idx)
                        self.idx2word[len(self.word2idx)-1] = token

                    context_list.append(self.word2idx[token])

                    for char_token in token:
                        if char_token not in self.char2idx:
                            self.char2idx[char_token] = len(self.char2idx)
                            self.idx2char[len(self.char2idx)-1] = char_token

                        char_list.append(self.char2idx[char_token])

                    context_char_list.append(char_list)

                # qas
                for qas_idx in range(len(qas)):
                    question_list = list()
                    question_char_list = list()

                    question = qas[qas_idx]['question'].replace("''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(question)
                    qa_id = qas[qas_idx]['id']

                    for ques_token in ques_tokens:
                        char_list = list()
                        # make question into word tokens
                        if ques_token not in self.word2idx:
                            self.word2idx[ques_token] = len(self.word2idx)
                            self.idx2word[len(self.word2idx)-1] = ques_token

                        question_list.append(self.word2idx[ques_token])
                        # make question into character tokens
                        for char_token in ques_token:
                            if char_token not in self.char2idx:
                                self.char2idx[char_token] = len(self.char2idx)
                                self.idx2char[len(self.char2idx)-1] = char_token

                            char_list.append(self.char2idx[char_token])

                        question_char_list.append(char_list)

                    for ans_idx in range(len(qas[qas_idx]['answers'])):
                        answer_list = list()
                        answer = qas[qas_idx]['answers'][ans_idx]['text']
                        ans_tokens = word_tokenize(answer)

                        # make answer label
                        for ans_token in ans_tokens:
                            if ans_token not in self.word2idx:
                                self.word2idx[ans_token] = len(self.word2idx)
                                self.idx2word[len(self.word2idx)-1] = ans_token

                            answer_list.append(self.word2idx[ans_token])

                        answer_start = _seq2seq(answer_list, context_list)
                        answer_stop  = answer_start + len(answer_list)

                        if answer_start == -1:
                            answer_mismatch += 1
                            break

                        self.train_data[train_idx] = dict()
                        self.dev_data[dev_idx] = dict()

                        # --------------- make question and answer pair --------------- #
                        if 'train' in self.config.mode:

                            self.train_data[train_idx]['question'] = question_list  # make train data with question index
                            self.train_data[train_idx]['context'] = context_list    # make train data with context index
                            self.train_data[train_idx]['question_char'] = question_char_list
                            self.train_data[train_idx]['context_char'] = context_char_list
                            self.train_data[train_idx]['answer_start'] = answer_start
                            self.train_data[train_idx]['answer_stop'] = answer_stop
                            self.train_data[train_idx]['id'] = qa_id
                            train_idx += 1

                        elif 'dev' in self.config.mode:
                            self.dev_data[dev_idx]['question'] = question_list  # make train data with question index
                            self.dev_data[dev_idx]['context'] = context_list    # make train data with context index
                            self.dev_data[dev_idx]['question_char'] = question_char_list
                            self.dev_data[dev_idx]['context_char'] = context_char_list
                            self.dev_data[dev_idx]['answer_start'] = answer_start
                            self.dev_data[dev_idx]['answer_stop'] = answer_stop
                            self.dev_data[dev_idx]['id'] = qa_id
                            dev_idx += 1

        print ("answer mismatch number",  answer_mismatch)

        # make glove table with word2idx
        self.word_idx2vec = np.zeros([len(self.word2idx), 300], dtype=np.float32)

        for word, idx in self.word2idx.items():
            if word == 'pad':
                continue
            else:
                try:
                    self.word_idx2vec[idx, :] = self.glove_file[word]
                except KeyError as e:
                    self.word_idx2vec[idx, :] = self.glove_file['unk']

        print ('char2idx number: ', len(self.char2idx))

    def _max_num_find(self, file_name):
        ### search max number of each dataset to use at max padding
        file_name = [file_name[i] for i in range(len(file_name)) if len(file_name[i]['context']) <= self.config.max_cont]
        # stop_answer = 0
        #
        # for idx in range(len(file_name)):
        #     if file_name[idx]['answer_stop'] > self.config.max_cont:
        #         stop_answer += 1
        #
        # print (stop_answer)

        return file_name

    def _make_numpy(self, data_name=None):
        ### make numpy data & max padding in question, context and answer
        ans_start, ans_stop, qa_id = [], [], []
        # batch_size, word, word_dim
        # batch_size, word, char, char_dim

        ques_mat = np.zeros((len(data_name), self.config.max_ques), dtype=np.int32)
        cont_mat = np.zeros((len(data_name), self.config.max_cont), dtype=np.int32)
        ques_char_mat = np.zeros((len(data_name), self.config.max_ques, self.config.max_ques_char), dtype=np.int32)
        cont_char_mat = np.zeros((len(data_name), self.config.max_cont, self.config.max_cont_char), dtype=np.int32)

        print ('question_matrix shape: ', ques_mat.shape)
        print ('context_matrix shape: ', cont_mat.shape)
        print ('question_char_matrix shape: ', ques_char_mat.shape)
        print ('context_char_matrix shape: ', cont_char_mat.shape)

        for idx in range(len(data_name)):
            ques_mat[idx, :len(data_name[idx]['question'][:self.config.max_ques])] = data_name[idx]['question'][:self.config.max_ques]
            cont_mat[idx, :len(data_name[idx]['context'][:self.config.max_cont])] = data_name[idx]['context'][:self.config.max_cont]

            for ques_idx in range(len(data_name[idx]['question_char'])):
                if ques_idx >= self.config.max_ques:
                    break
                new_ques = data_name[idx]['question_char'][ques_idx]
                ques_char_mat[idx, ques_idx, :len(new_ques)] = new_ques

            for cont_idx in range(len(data_name[idx]['context_char'])):
                if cont_idx >= self.config.max_cont:
                    break
                new_cont = data_name[idx]['context_char'][cont_idx]
                cont_char_mat[idx, cont_idx, :len(new_cont)] = new_cont

            ans_start.append(data_name[idx]['answer_start'])
            ans_stop.append(data_name[idx]['answer_stop'])
            qa_id.append(data_name[idx]['id'])

        reduced_size = len(data_name) // self.config.batch_size

        ques_mat = ques_mat[:reduced_size * self.config.batch_size]
        cont_mat = cont_mat[:reduced_size * self.config.batch_size]
        ques_char_mat = ques_char_mat[:reduced_size * self.config.batch_size]
        cont_char_mat = cont_char_mat[:reduced_size * self.config.batch_size]
        ans_start = np.array(ans_start[:reduced_size * self.config.batch_size], dtype=np.int32)
        ans_stop  = np.array(ans_stop[:reduced_size * self.config.batch_size], dtype=np.int32)
        qa_id = np.array(qa_id[:reduced_size * self.config.batch_size])

        ans_start = np.eye(self.config.max_cont)[ans_start]
        ans_stop  = np.eye(self.config.max_cont)[ans_stop]

        ### make input data as batch shape
        self.ques_mat = np.reshape(ques_mat, [reduced_size, self.config.batch_size, -1])
        self.cont_mat = np.reshape(cont_mat, [reduced_size, self.config.batch_size, -1])
        self.ques_char_mat = np.reshape(ques_char_mat, [reduced_size, self.config.batch_size, self.config.max_ques, self.config.max_ques_char])
        self.cont_char_mat = np.reshape(cont_char_mat, [reduced_size, self.config.batch_size, self.config.max_cont, self.config.max_cont_char])
        self.ans_start = np.reshape(ans_start, [reduced_size, self.config.batch_size, self.config.max_cont])
        self.ans_stop = np.reshape(ans_stop, [reduced_size, self.config.batch_size, self.config.max_cont])
        self.qa_id = np.reshape(qa_id, [reduced_size, self.config.batch_size])

        print ()
        print ('question_matrix shape: ', self.ques_mat.shape)
        print ('context_matrix shape: ', self.cont_mat.shape)
        print ('question_char_matrix shape: ', self.ques_char_mat.shape)
        print ('context_char_matrix shape: ', self.cont_char_mat.shape)
        print ('answer start matrix shape: ', self.ans_start.shape)
        print ('answer stop matrix shape: ', self.ans_stop.shape)
        print ('number of question id: ', self.qa_id.shape)

        ques_mat = list(self.ques_mat)
        cont_mat = list(self.cont_mat)
        ques_char_mat = list(self.ques_char_mat)
        cont_char_mat = list(self.cont_char_mat)
        ans_start = list(self.ans_start)
        ans_stop = list(self.ans_stop)
        qa_id = list(self.qa_id)

        self.zip_list = list(zip(ques_mat, cont_mat, ques_char_mat, cont_char_mat, ans_start, ans_stop, qa_id))

        if 'train' in self.config.mode:
            with open('tr_zip_list.pkl', 'wb') as fp:
                pickle.dump(self.zip_list, fp)
        elif 'dev' in self.config.mode:
            with open('dev_zip_list.pkl', 'wb') as fp:
                pickle.dump(self.zip_list, fp)

    def _iter(self, ):
        ### iteration used at train.py
        for ques, cont, ques_char, cont_char, ans_start, ans_stop, qa_id in self.zip_list:
            yield ques, cont, ques_char, cont_char, ans_start, ans_stop, qa_id

def main():
    config = get_args()
    dataset = Squad_Dataset(config)

if __name__ == "__main__":
    main()

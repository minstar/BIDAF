import os
import re
import json
import pickle
import pprint
import numpy as np

# for sanity check
from config import get_args

class Squad_Dataset():
    def __init__(self, config):
        self.config = config
        self.word2idx = dict()
        self.idx2word = dict()
        self.char2idx = dict()
        self.idx2char = dict()
        self.train_data = dict()

        self._load()
        self._read()
        self._make_numpy()
        self._get_batch()

    def clean_str(self, text):
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
        text = re.sub(r'\{', ' {', text)
        text = re.sub(r'\}', ' }', text)
        text = re.sub(r'\<', ' <', text)
        text = re.sub(r'\>', ' >', text)
        return text

    # load glove and squad data
    def _load(self, ):
        with open(self.config.train_dir + self.config.train_file, 'r') as fp:
            self.train_file = json.load(fp)['data']
        with open(self.config.train_dir + self.config.dev_file, 'r') as fp:
            self.dev_file = json.load(fp)['data']

    def _read(self, ):

        train_idx = 0
        print (" # --------------- start making training data --------------- # \n")
        self.word2idx['pad'] = 0
        self.idx2word[0] = 'pad'
        self.word2idx['unk'] = 1
        self.idx2word[1] = 'unk'

        for idx in range(len(self.train_file)):
            paragraphs = self.train_file[idx]['paragraphs']
            for par_idx, paragraph in enumerate(paragraphs):
                context_list = list()
                context_char_list = list()

                context = self.clean_str(paragraph['context'].lower())
                qas = paragraph['qas']
                tokens = context.split()

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
                    self.train_data[train_idx] = dict()
                    question_list = list()
                    question_char_list = list()
                    answer_list = list()

                    question = self.clean_str(qas[qas_idx]['question'].lower())
                    answer = self.clean_str(qas[qas_idx]['answers'][0]['text'].lower())
                    answer_start = qas[qas_idx]['answers'][0]['answer_start']
                    answer_stop = answer_start + len(answer)

                    ques_tokens = question.split()
                    ans_tokens = answer.split()

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

                    for ans_token in ans_tokens:
                        if ans_token not in self.word2idx:
                            # print ("# --------------- answer is not in the context ---------------# \n")
                            self.word2idx[ans_token] = len(self.word2idx)
                            self.idx2word[len(self.word2idx)-1] = ans_token

                        answer_list.append(self.word2idx[ans_token])

                    # --------------- make question and answer pair --------------- #
                    self.train_data[train_idx]['question'] = question_list  # make train data with question index
                    self.train_data[train_idx]['context'] = context_list    # make train data with context index
                    self.train_data[train_idx]['answer'] = answer_list      # make train data with answer index
                    self.train_data[train_idx]['question_char'] = question_char_list
                    self.train_data[train_idx]['context_char'] = context_char_list
                    self.train_data[train_idx]['answer_start'] = answer_start
                    self.train_data[train_idx]['answer_stop'] = answer_stop
                    train_idx += 1

                # make fake answer-question pair data
                # TODO

    ### search max number of each dataset to use at max padding
    def _max_num_find(self,):
        max_question = 0
        max_context = 0
        max_context_char = 0
        max_question_char = 0

        for idx in range(len(self.train_data)):

            if max_question < len(self.train_data[idx]['question']):
                max_question = len(self.train_data[idx]['question'])

            if max_context < len(self.train_data[idx]['context']):
                max_context = len(self.train_data[idx]['context'])

            for context_idx in range(len(self.train_data[idx]['context_char'])):
                if max_context_char < len(self.train_data[idx]['context_char'][context_idx]):
                    max_context_char = len(self.train_data[idx]['context_char'][context_idx])

            for ques_idx in range(len(self.train_data[idx]['question_char'])):
                if max_question_char < len(self.train_data[idx]['question_char'][ques_idx]):
                    max_question_char = len(self.train_data[idx]['question_char'][ques_idx])

    ### make numpy data & max padding in question, context and answer
    def _make_numpy(self,):

        ans_start = list()
        ans_stop  = list()
        # batch_size, word, word_dim
        # batch_size, word, char, char_dim
        ques_mat = np.zeros((len(self.train_data), self.config.max_ques), dtype=np.int32)
        cont_mat = np.zeros((len(self.train_data), self.config.max_cont), dtype=np.int32)
        ques_char_mat = np.zeros((len(self.train_data), self.config.max_ques, self.config.max_ques_char), dtype=np.int32)
        cont_char_mat = np.zeros((len(self.train_data), self.config.max_cont, self.config.max_cont_char), dtype=np.int32)

        print ('question_matrix shape: ', ques_mat.shape)
        print ('context_matrix shape: ', cont_mat.shape)
        print ('question_char_matrix shape: ', ques_char_mat.shape)
        print ('context_char_matrix shape: ', cont_char_mat.shape)

        for idx in range(len(self.train_data)):
            ques_mat[idx, :len(self.train_data[idx]['question'])] = self.train_data[idx]['question']
            cont_mat[idx, :len(self.train_data[idx]['context'])] = self.train_data[idx]['context']

            for ques_idx in range(len(self.train_data[idx]['question_char'])):
                new_ques = self.train_data[idx]['question_char'][ques_idx]
                ques_char_mat[idx, ques_idx, :len(new_ques)] = new_ques

            for cont_idx in range(len(self.train_data[idx]['context_char'])):
                new_cont = self.train_data[idx]['context_char'][cont_idx]
                cont_char_mat[idx, cont_idx, :len(new_cont)] = new_cont

            ans_start.append(self.train_data[idx]['answer_start'])
            ans_stop.append(self.train_data[idx]['answer_stop'])

        reduced_size = len(self.train_data) // self.config.batch_size

        ques_mat = ques_mat[:reduced_size * self.config.batch_size]
        cont_mat = cont_mat[:reduced_size * self.config.batch_size]
        ques_char_mat = ques_char_mat[:reduced_size * self.config.batch_size]
        cont_char_mat = cont_char_mat[:reduced_size * self.config.batch_size]
        ans_start = np.array(ans_start[:reduced_size * self.config.batch_size], dtype=np.int32)
        ans_stop  = np.array(ans_stop[:reduced_size * self.config.batch_size], dtype=np.int32)

        ### make input data as batch shape
        self.ques_mat = np.reshape(ques_mat, [reduced_size, self.config.batch_size, -1])
        self.cont_mat = np.reshape(cont_mat, [reduced_size, self.config.batch_size, -1])
        self.ques_char_mat = np.reshape(ques_char_mat, [reduced_size, self.config.batch_size, self.config.max_ques, self.config.max_ques_char])
        self.cont_char_mat = np.reshape(cont_char_mat, [reduced_size, self.config.batch_size, self.config.max_cont, self.config.max_cont_char])
        self.ans_start = np.reshape(ans_start, [reduced_size, self.config.batch_size])
        self.ans_stop = np.reshape(ans_stop, [reduced_size, self.config.batch_size])

        print ()
        print ('question_matrix shape: ', self.ques_mat.shape)
        print ('context_matrix shape: ', self.cont_mat.shape)
        print ('question_char_matrix shape: ', self.ques_char_mat.shape)
        print ('context_char_matrix shape: ', self.cont_char_mat.shape)
        print ('answer start matrix shape: ', self.ans_start.shape)
        print ('answer stop matrix shape: ', self.ans_stop.shape)

    ### get batch dataset
    def _get_batch(self, ):
        self.ques_mat = list(self.ques_mat)
        self.cont_mat = list(self.cont_mat)
        self.ques_char_mat = list(self.ques_char_mat)
        self.cont_char_mat = list(self.cont_char_mat)
        self.ans_start = list(self.ans_start)
        self.ans_stop = list(self.ans_stop)

        self.zip_list = list(zip(self.ques_mat, self.cont_mat, self.ques_char_mat, self.cont_char_mat, self.ans_start, self.ans_stop))

    def _iter(self, ):
        for ques, cont, ques_char, cont_char, ans_start, ans_stop in self.zip_list:
            yield ques, cont, ques_char, cont_char, ans_start, ans_stop

def main():
    config = get_args()
    dataset = Squad_Dataset(config)

if __name__ == "__main__":
    main()

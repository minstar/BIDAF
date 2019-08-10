import pdb
import pickle
import tensorflow as tf
import numpy as np

from model import BIDAF
from preprocess import Squad_Dataset
from config import get_args

class Trainer():
    def __init__(self, config, data, model, saver, sess):
        self.config = config
        self.data = data
        self.model = model
        self.sess = sess
        self.saver = saver

        if self.config.mode == 'dev': self._load_data()
        self.loss = self.model.loss
        self.train_opt = self.model.train_opt
        self.global_step = self.model.global_step
        self.zip_list = self.data.zip_list

    def _load_data(self, ):
        with open('dev_zip_list.pkl', 'rb') as fp:
            self.dev_zip_list = pickle.load(fp)

    def train(self, ):

        if self.config.mode == 'train':
            tr_ema, tr_loss = [], []
            for epoch in range(self.config.epochs):
                print (" -------------------- Epoch %d is ongoing -------------------- \n" % (epoch))
                for train_idx, (ques, cont, ques_char, cont_char, ans_start, ans_stop, qa_id) in enumerate(self.zip_list):
                    loss, global_step, pred1, pred2 = self.train_step(ques, cont, ques_char, cont_char, ans_start, ans_stop)

                    tr_loss.append(np.mean(loss))

                    # tr_ema.append(ema)

                    if (train_idx+1) % self.config.print_step == 0:
                        print ('Epoch %d, train_step %d: loss : %.4f \n' % (epoch, train_idx, np.mean(loss)))

                self.saver.save(self.sess, self.config.save_dir)
                print ('Successfully saved model\n')

        elif self.config.mode == 'dev':
            pred_dict = dict()
            for dev_idx, (ques, cont, ques_char, cont_char, ans_start, ans_stop, qa_id) in enumerate(self.dev_zip_list):
                loss, global_step, pred1, pred2 = self.evaluate(ques, cont, ques_char, cont_char, ans_start, ans_stop)
                start_1, start_2 = np.where(pred1 == 1)
                stop_1, stop_2 = np.where(pred2 == 1)

                for index_idx in start_1:
                    answer_str = ''
                    need_decode = cont_mat[index_idx][start_2[index_idx]:stop_2[index_idx]]
                    for dec_idx in need_decode:
                        answer_str += ''.join(data.idx2word[dec_idx])
                        answer_str += ' '

                    pred_dict[qa_id[index_idx]] = answer_str[:-1]

            with open('./predictions.json', 'w', encoding='utf-8') as fp:
                json.dump(pred_dict, fp, indent=2)

    def evaluate(self, ques, cont, ques_char, cont_char, ans_start, ans_stop):
        feed_dict = self.create_feed_dict(ques, cont, ques_char, cont_char, ans_start, ans_stop)
        _, loss, global_step, pred1, pred2= self.sess.run([self.model.train_opt, self.model.loss, self.global_step, \
                                                self.model.pred1, self.model.pred2], feed_dict=feed_dict)

        return loss, global_step, pred1, pred2


    def train_step(self, ques, cont, ques_char, cont_char, ans_start, ans_stop):
        feed_dict = self.create_feed_dict(ques, cont, ques_char, cont_char, ans_start, ans_stop)
        _, loss , global_step, pred1, pred2 = self.sess.run([self.model.train_opt, self.model.loss, self.global_step, \
                                                        self.model.pred1, self.model.pred2], feed_dict=feed_dict)

        return loss, global_step, pred1, pred2

    def create_feed_dict(self, ques, cont, ques_char, cont_char, ans_start, ans_stop):
        if self.config.mode == 'train':
            feed_dict = {self.model.ques_word:ques,
                        self.model.cont_word:cont,
                        self.model.ques_char:ques_char,
                        self.model.cont_char:cont_char,
                        self.model.answer_start:ans_start,
                        self.model.answer_stop:ans_stop}
        elif self.config.mode == 'dev':
            feed_dict = {self.model.ques_word:ques,
                        self.model.cont_word:cont,
                        self.model.ques_char:ques_char,
                        self.model.cont_char:cont_char,
                        self.model.answer_start:ans_start,
                        self.model.answer_stop:ans_stop}
        elif self.config.mode == 'test':
            pass

        return feed_dict

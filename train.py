import pdb
import json
import pickle
import tensorflow as tf
import numpy as np

from model import BIDAF
from preprocess import Squad_Dataset
from config import get_args
from evaluator import *

class Trainer():
    def __init__(self, config, data, model, saver, sess, exp_name, writer):
        self.config = config
        self.data = data
        self.model = model
        self.sess = sess
        self.saver = saver
        self.exp_name = exp_name
        self.writer = writer

        self.loss = self.model.loss
        self.train_opt = self.model.train_opt
        self.global_step = self.model.global_step

    def train(self, ):

        tr_ema, tr_loss = [], []
        for epoch in range(self.config.epochs):

            print (" -------------------- Epoch %d is ongoing -------------------- \n" % (epoch))
            for train_idx, (ques, cont_mat, ques_char, cont_char, ans_start, ans_stop, qa_id) in enumerate(self.data.tr_zip_list):
                # loss, global_step, ce, arg_p1, arg_p2 = self.train_step(ques, cont_mat, ques_char, cont_char, ans_start, ans_stop)
                loss = self.train_step(ques, cont_mat, ques_char, cont_char, ans_start, ans_stop)
                # tr_loss.append(ce)

                if (train_idx+1) % self.config.print_step == 0:
                    print ('Epoch %d, train_step %d: loss %.4f \n' % (epoch, train_idx, loss))

            self.saver.save(self.sess, "%s/%s" % (self.config.save_dir, self.exp_name))
            print ('Successfully saved model\n')

            pred_dict = dict()
            for dev_idx, (ques, cont_mat, ques_char, cont_char, ans_start, ans_stop, qa_id) in enumerate(self.data.dev_zip_list):
                loss, global_step, arg_p1, arg_p2 = self.evaluate(ques, cont_mat, ques_char, cont_char, ans_start, ans_stop)
                pred1_lst = arg_p1.tolist()
                pred2_lst = arg_p2.tolist()
                for index_idx in range(len(pred1_lst)):
                    answer_str = ''
                    need_decode = cont_mat[index_idx][pred1_lst[index_idx]:pred2_lst[index_idx]]
                    for dec_idx in need_decode:
                        answer_str += ''.join(self.data.idx2word[dec_idx])
                        answer_str += ' '

                    if answer_str == '':
                        pred_dict[qa_id[index_idx]] = ''
                    else:
                        pred_dict[qa_id[index_idx]] = answer_str[:-1]

                if (dev_idx+1) % self.config.print_step == 0:
                    print ('Epoch %d, dev_step %d \n' % (epoch, dev_idx))

            results = evaluate(self.data.dev_file, pred_dict)
            with open('./out/predictions_%d.json' % (epoch), 'w', encoding='utf-8') as fp:
                json.dump(pred_dict, fp, indent=2)
            with open('./out/results_%d.json' % (epoch), 'w', encoding='utf-8') as fp:
                json.dump(results, fp)

        if self.config.mode == 'test':
            pass

    def evaluate(self, ques, cont_mat, ques_char, cont_char, ans_start, ans_stop):
        feed_dict = self.create_feed_dict(ques, cont_mat, ques_char, cont_char, ans_start, ans_stop)
        _, loss, global_step, prob1, prob2 = self.sess.run([self.train_opt, self.loss, self.global_step, \
                                                                self.model.arg_p1, self.model.arg_p2], feed_dict=feed_dict)

        return loss, global_step, prob1, prob2


    def train_step(self, ques, cont_mat, ques_char, cont_char, ans_start, ans_stop):
        feed_dict = self.create_feed_dict(ques, cont_mat, ques_char, cont_char, ans_start, ans_stop)
        loss, _, global_step = self.sess.run([self.loss, self.train_opt, self.global_step], feed_dict=feed_dict)
        summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss)])
        self.writer.add_summary(summary, global_step)
        return loss
        # _, loss, global_step, ce, prob1, prob2 = self.sess.run([self.train_opt, self.loss, self.global_step, \
        #                                                         self.model.cross_entropy, self.model.arg_p1, self.model.arg_p2], feed_dict=feed_dict)

        # return loss, global_step, ce, prob1, prob2, pr1, pr2

    def create_feed_dict(self, ques, cont_mat, ques_char, cont_char, ans_start, ans_stop):
        if self.config.mode == 'train':
            feed_dict = {self.model.ques_word:ques,
                        self.model.cont_word:cont_mat,
                        self.model.ques_char:ques_char,
                        self.model.cont_char:cont_char,
                        self.model.answer_start:ans_start,
                        self.model.answer_stop:ans_stop}
        elif self.config.mode == 'dev':
            feed_dict = {self.model.ques_word:ques,
                        self.model.cont_word:cont_mat,
                        self.model.ques_char:ques_char,
                        self.model.cont_char:cont_char,
                        self.model.answer_start:ans_start,
                        self.model.answer_stop:ans_stop}
        elif self.config.mode == 'test':
            pass

        return feed_dict

import pdb
import tensorflow as tf
import numpy as np

from model import BIDAF
from preprocess import Squad_Dataset
from config import get_args

class Trainer():
    def __init__(self, config, data, model, sess):
        self.config = config
        self.data = data
        self.model = model
        self.sess = sess

        self.loss = self.model.loss
        self.train_opt = self.model.train_opt
        self.global_step = self.model.global_step
        self.zip_list = self.data.zip_list

    def train(self, ):
        tr_ema, tr_loss = [], []

        for epoch in range(self.config.epochs):
            print (" -------------------- Epoch %d is ongoing -------------------- \n")
            for train_idx, (ques, cont, ques_char, cont_char, ans_start, ans_stop) in enumerate(self.zip_list):
                loss, global_step = self.train_step(ques, cont, ques_char, cont_char, ans_start, ans_stop)

                tr_loss.append(np.mean(loss))
                # tr_ema.append(ema)

                if (train_idx+1) % self.config.print_step == 0:
                    print ('Epoch %d, train_step %d: loss : %.4f \n' % (epoch, train_idx, np.mean(loss)))

    def train_step(self, ques, cont, ques_char, cont_char, ans_start, ans_stop):
        feed_dict = self.create_feed_dict(ques, cont, ques_char, cont_char, ans_start, ans_stop)
        _, loss , global_step = self.sess.run([self.model.train_opt, self.model.loss, self.global_step], \
                                                    feed_dict=feed_dict)

        return loss, global_step

    def create_feed_dict(self, ques, cont, ques_char, cont_char, ans_start, ans_stop):
        if self.config.mode == 'train':
            feed_dict = {self.model.ques_word:ques,
                        self.model.cont_word:cont,
                        self.model.ques_char:ques_char,
                        self.model.cont_char:cont_char,
                        self.model.answer_start:ans_start,
                        self.model.answer_stop:ans_stop}
        elif self.config.mode == 'dev':
            pass
        elif self.config.mode == 'test':
            pass

        return feed_dict

import numpy as np
import os
import time
import pdb
import pickle
import pprint
import json
import re
import time
import random
import string

import tensorflow as tf
from collections import Counter

from sklearn.utils import shuffle
from preprocess import Squad_Dataset
from config import get_args

def main():
    config = get_args()
    data = Squad_Dataset(config)
    with open('dev_zip_list.pkl', 'rb') as fp:
        dev_zip_list = pickle.load(fp)

    pred_dict = dict()
    for idx, (ques_mat, cont_mat, ques_char_mat, cont_char_mat, ans_start, ans_stop, qa_id) in enumerate(dev_zip_list):
        start_1, start_2 = np.where(ans_start == 1)
        stop_1, stop_2 = np.where(ans_stop == 1)

        for index_idx in start_1:
            answer_str = ''
            need_decode = cont_mat[index_idx][start_2[index_idx]:stop_2[index_idx]]
            for dec_idx in need_decode:
                answer_str += ''.join(data.idx2word[dec_idx])
                answer_str += ' '

            pred_dict[qa_id[index_idx]] = answer_str[:-1]

    with open('./predictions.json', 'w', encoding='utf-8') as fp:
        json.dump(pred_dict, fp, indent=2)

if __name__ == "__main__":
    main()

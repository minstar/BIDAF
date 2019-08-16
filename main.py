import os, pdb
import random
import numpy as np
import tensorflow as tf

from model import BIDAF
from train import Trainer
from config import get_args
from evaluator import *
from preprocess import Squad_Dataset


def main():
    config = get_args()
    dataset = Squad_Dataset(config)
    model = BIDAF(config, dataset.word_idx2vec)
    config.mode = 'train'

    # make run_config
    run_config = tf.ConfigProto(log_device_placement=False)
    run_config.gpu_options.allow_growth = True
    exp_name = '%s_%s_%s' % (config.train_file, config.glove_file, config.lr)

    # save train file
    if not (os.path.exists(os.path.join(config.save_dir, exp_name))):
        os.makedirs(os.path.join(config.save_dir, exp_name))

    sess = tf.Session(config=run_config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    
    # model.model_summary()
    writer = tf.summary.FileWriter('./logs/', sess.graph)
    writer.add_graph(sess.graph)
    loader = tf.train.Saver(max_to_keep=5)

    if config.mode == 'train':
        trainer = Trainer(config, dataset, model, loader, sess, exp_name, writer)
        trainer.train()

    elif config.mode == 'test':
        # load best trainer for testing
        print ("restore latest evaluation model\n")
        loader.restore(sess, tf.train.latest_checkpoint(os.path.join(config.save_dir, exp_name)))
        trainer = Trainer(config, dataset, model, loader, sess, exp_name, writer)
        trainer.train()

if __name__ == "__main__":
    main()

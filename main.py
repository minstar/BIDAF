import os, pdb
import numpy as np
import tensorflow as tf

from train import Trainer
from config import get_args
from preprocess import Squad_Dataset
from model import BIDAF

def main():
    config = get_args()
    dataset = Squad_Dataset(config)
    model = BIDAF(config, dataset.word_idx2vec)

    # make run_config
    run_config = tf.ConfigProto(log_device_placement=False)
    run_config.gpu_options.allow_growth = True
    exp_name = '%s_%s_%s_%s' % (config.mode, config.train_file, config.glove_file, config.lr)

    # save train file
    if not (os.path.exists(os.path.join(config.save_dir, exp_name))):
        os.makedirs(os.path.join(config.save_dir, exp_name))

    sess = tf.Session(config=run_config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    model.model_summary()
    loader = tf.train.Saver(max_to_keep=None)

    if config.mode == 'train':
        trainer = Trainer(config, dataset, model, loader, sess)
        # evaluator = Evaluator(config, dataset, model, sess)

    trainer.train()

    # load best trainer during evaluation
    print ("load latest evaluation model\n")
    # loader.restore(sess, tf.train.latest_checkpoint(os.path.join(config.save_dir, exp_name)))

    # needs evaluation code

if __name__ == "__main__":
    main()

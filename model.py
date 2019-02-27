import tensorflow as tf

from config import *

# word_embedding
class word_embedding:
    def __init__(self, word_idx2vec):
        self.word_idx2vec = word_idx2vec

    def text_word_embedding(self, scope=None):
        word_text_input = tf.placeholder(tf.int32, [FLAGS.max_text])

        with tf.variable_scope(scope):
            glove_table = tf.get_variable('glove_table', initializer=self.word_idx2vec, trainable=False)
            self.text_glove = tf.nn.embedding_lookup(glove_table, word_text_input)

        return self.text_glove # (677, 300)

    def query_word_embedding(self, scope=None):
        word_query_input = tf.placeholder(tf.int32, [FLAGS.max_num_query, FLAGS.max_query])

        with tf.variable_scope(scope):
            glove_table = tf.get_variable('glove_table', initializer=self.word_idx2vec, trainable=False)
            self.query_glove = tf.nn.embedding_lookup(glove_table, word_query_input)

        return self.query_glove # (31, 53, 300)

# character_embedding
def conv2d(input_char, filter_num, filter_width, name="conv2d"):
    # --------------------------- Input --------------------------- #
    # input_char : shape of (max_text, word_maxlen, char_dimension)
    # filter_num : [25, 50, 75, 100, 125, 150] subscribed in paper
    # filter_width : [1,2,3,4,5,6] subscribed in paper.

    # --------------------------- Output --------------------------- #
    # convolution output, filtered with each kernel size and width
    with tf.variable_scope(name):
        w = tf.get_variable(name="filters", shape=[1, filter_width, FLAGS.char_dimension, filter_num])
        b = tf.get_variable(name='filters_bias', shape=[filter_num])

    return tf.nn.conv2d(input_char, w, strides=[1,1,1,1], padding='VALID') + b

def TDNN(input_embedded, scope='TDNN'):
    # --------------------------- Input --------------------------- #
    # input_embedded : shape of (max_text, word_maxlen, char_dimension)
    # example : (677, 23, 15)

    # --------------------------- Output --------------------------- #
    # output : Concatenated version of max-over-time pooling layer

    word_maxlen = input_embedded.get_shape()[1]

    # make channel as 1 with expand_dim
    input_char = tf.expand_dims(input_embedded, axis=1) # (677, 1, 23, 15)

    layers = list()
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(FLAGS.kernel_width, FLAGS.kernel_features):

            conv = conv2d(input_char, kernel_feature_size, kernel_size, name="kernel_%d" % kernel_size)

            pool = tf.nn.max_pool(tf.tanh(conv), ksize=(1,1,word_maxlen-kernel_size + 1,1), strides=[1,1,1,1], padding='VALID')

            # get feature map when needed
            layers.append(tf.squeeze(pool, axis=[1,2]))

        output = tf.concat(layers, axis=1)

    return output

class char_embedding:
    def __init__(self, word_maxlen, char_size=124):
        self.word_maxlen = word_maxlen
        self.char_size = char_size

    def text_char_embedding(self, scope=None):
        char_text_input = tf.placeholder(tf.int32, [FLAGS.max_text, self.word_maxlen])

        with tf.variable_scope(scope):
            char_embedding = tf.get_variable('char_text_emb', shape=[self.char_size, FLAGS.char_dimension])

            # first row zero padding, if needed
            char_embedding_padded = tf.scatter_update(ref=char_embedding, indices=[0], \
                                                     updates=tf.constant(0.0, tf.float32, [1, FLAGS.char_dimension]))

            self.input_embedded = tf.nn.embedding_lookup(char_embedding, char_text_input) # (677, 23, 15)

            self.output_embedded = TDNN(self.input_embedded, scope='TDNN')

        return self.output_embedded # (677, 525)

    def query_char_embedding(self, scope=None):
        char_text_input = tf.placeholder(tf.int32, [FLAGS.max_num_query, FLAGS.max_query, self.word_maxlen])

        with tf.variable_scope(scope):
            char_embedding = tf.get_variable('char_text_emb', shape=[self.char_size, FLAGS.char_dimension])

            # first row zero padding, if needed
            char_embedding_padded = tf.scatter_update(ref=char_embedding, indices=[0], \
                                                     updates=tf.constant(0.0, tf.float32, [1, FLAGS.char_dimension]))

            self.input_embedded = tf.nn.embedding_lookup(char_embedding, char_text_input)

            self.output_embedded = TDNN(self.input_embedded, scope='TDNN')

        return self.output_embedded

# contextual embedding
def lstm_cell():
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.LSTM_hidden, state_is_tuple=True, forget_bias=0.0, reuse=False)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.LSTM_hidden, state_is_tuple=True, forget_bias=0.0, reuse=False)

    if FLAGS.dropout > 0.0:
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=1.0 - FLAGS.dropout)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=1.0 - FLAGS.dropout)

    return lstm_fw_cell, lstm_bw_cell

def context_embedding():



# Attention Flow layer
# Modeling layer
# Multi layer Perceptron
# assemble graph
# training operation
# loss operation

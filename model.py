import tensorflow as tf
import pdb

from config import get_args
from preprocess import Squad_Dataset

class BIDAF():
    def __init__(self, config, word_idx2vec):
        self.config = config
        self.word_idx2vec = word_idx2vec
        self.keep_prob = 1 - self.config.dropout

        self.build_model()

    def word_embedding(self, word_input, scope=None):
        ### make word embedding of context or question.
        ### word_input : context word or question word
        with tf.variable_scope(scope):
            glove_table = tf.get_variable('glove_table', initializer=self.word_idx2vec, trainable=False)
            glove_embed = tf.nn.embedding_lookup(glove_table, word_input)

        return glove_embed

    def char_embedding(self, char_input, is_query=None, scope=None):
        ### make character embedding of context or question.
        ### char_input : context character or question character
        with tf.variable_scope(scope):
            char_embed = tf.get_variable('char_embed', shape=[self.config.char_6b, self.config.char_dim]) # for convenience of training time.
            # zero padding
            char_embed_pad = tf.scatter_update(ref=char_embed, indices=[0], \
                                                updates=tf.constant(0.0, tf.float32, [1, self.config.char_dim]))

            input_embed = tf.nn.embedding_lookup(char_embed, char_input)
            output_embed = self.TDNN(input_embed, is_query, scope='TDNN')
        return output_embed

    def conv2d(self, input_char, filter_width, filter_num, scope=None):
        with tf.variable_scope(scope):
            w = tf.get_variable(name="filters", shape=[1, filter_width, self.config.char_dim, filter_num]) # (1, 5, 20, 100)
            b = tf.get_variable(name='filters_bias', shape=[filter_num])

        return tf.nn.conv2d(input_char, w, strides=[1,1,1,1], padding='VALID') + b

    def TDNN(self, input_embed, is_query=True, scope=None):
        layers = list()
        with tf.variable_scope(scope):
            for kernel_size, kernel_feature_size in zip(self.config.filter_width, self.config.filter_num):
                conv = self.conv2d(input_embed, kernel_size, kernel_feature_size, scope="filter_%d"%kernel_size)
                if is_query:
                    # question
                    pool = tf.nn.max_pool(tf.tanh(conv), ksize=(1,1,self.config.max_ques_char-kernel_size+1,1), strides=[1,1,1,1], padding='VALID')
                else:
                    pool = tf.nn.max_pool(tf.tanh(conv), ksize=(1,1,self.config.max_cont_char-kernel_size+1,1), strides=[1,1,1,1], padding='VALID')

                layers.append(tf.squeeze(pool, axis=[2]))

            output = tf.concat(layers, axis=2)
            return output

    def highway(self, input_highway, bias=-2.0, scope=None):
        ### Highway network to get interpolation result
        output_dim = input_highway.get_shape()[2]
        input_highway = tf.reshape(input_highway, [-1, output_dim])

        with tf.variable_scope(scope):
            for i in range(self.config.highway_layers):
                t = tf.sigmoid(self.Affine_Transformation(input_highway, output_dim, scope='transgate_%d' % i) + bias)
                g = tf.nn.relu(self.Affine_Transformation(input_highway, output_dim, scope='MLP_%d' % i))

                z = t * g + (1.0 - t) * input_highway

                if self.config.highway_layers > 1:
                    ### convert next (t+1) highway input to (t) output
                    input_highway = z

            z = tf.reshape(z, [self.config.batch_size, -1, output_dim])
            return z

    def Affine_Transformation(self, input_highway, output_dim, scope=None):
        ### input_highway : word embedding + char embedding - (batch * max_context, dim_sum)
        with tf.variable_scope(scope):
            w = tf.get_variable(name="highway_matrix", shape=[output_dim, input_highway.get_shape()[1]], dtype=tf.float32)
            b = tf.get_variable(name="highway_bias", shape=[output_dim], dtype=tf.float32)

            return tf.matmul(input_highway, w) + b

    def context_embedding(self, context_input, scope=None):
        ### utilizing contextual cues in word embedding
        with tf.variable_scope(scope):
            fw_cells = [tf.nn.rnn_cell.LSTMCell(64, state_is_tuple=True) for _ in range(self.config.lstm_layers)]
            bw_cells = [tf.nn.rnn_cell.LSTMCell(64, state_is_tuple=True) for _ in range(self.config.lstm_layers)]

            fw_dropout = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob) for cell in fw_cells]
            bw_dropout = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob) for cell in bw_cells]

            fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_dropout, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_dropout, state_is_tuple=True)

            output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, context_input, time_major=True, dtype=tf.float32)

            # forward output and backward output
            # tf.concat((output[0], output[1]))
            pdb.set_trace()
            return output


    def attention_flow(self, ):
        pass

    def modeling_layer(self, ):
        pass

    def output_layer(self, ):
        pass

    def build_model(self, ):
        ### building total graph
        self.cont_word = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_cont])
        self.ques_word = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ques])
        self.cont_char = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_cont, self.config.max_cont_char])
        self.ques_char = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ques, self.config.max_ques_char])

        ### Word Embedding Layer
        cont_glove = self.word_embedding(self.cont_word, scope="context_word_emb")  # (32, 791, 300)
        ques_glove = self.word_embedding(self.ques_word, scope="question_word_emb") # (32, 60, 300)

        ### Character Embedding Layer
        cont_char = self.char_embedding(self.cont_char, is_query=False, scope="context_char_emb") # (32, 791, 525)
        ques_char = self.char_embedding(self.ques_char, is_query=True, scope="question_char_emb") # (32, 60, 525)

        ### make list to concat word embedding vector and character embedding vector
        cont_concat = tf.concat([cont_glove, cont_char], axis=2)  # (32, 791, 825)
        ques_concat = tf.concat([ques_glove, ques_char], axis=2)  # (32, 60, 825)

        ### Highway network get dimension d
        cont_highway = self.highway(cont_concat, bias=-2.0, scope="context_highway_net")
        ques_highway = self.highway(ques_concat, bias=-2.0, scope="question_highway_net")

        ### Contextual Embedding Layer
        cont_output = self.context_embedding(cont_highway, scope="context_cont")
        ques_output = self.context_embledding(ques_highway, scope="context_ques")

        ### Attention Flow Layer

        ### Modeling Layer

        ### Output Layer


# for sanity check
def main():
    config = get_args()
    dataset = Squad_Dataset(config)
    model = BIDAF(config, dataset.word_idx2vec)

if __name__ == "__main__":
    main()

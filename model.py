import pdb
import tensorflow as tf
import tensorflow.contrib.slim as slim

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
        # output_dim = input_highway.get_shape()[2]
        input_highway = tf.reshape(input_highway, [-1, self.output_dim])

        with tf.variable_scope(scope):
            for i in range(self.config.highway_layers):
                t = tf.sigmoid(self.Affine_Transformation(input_highway, self.output_dim, scope='transgate_%d' % i) + bias)
                g = tf.nn.relu(self.Affine_Transformation(input_highway, self.output_dim, scope='MLP_%d' % i))

                z = t * g + (1.0 - t) * input_highway

                if self.config.highway_layers > 1:
                    ### convert next (t+1) highway input to (t) output
                    input_highway = z

            z = tf.reshape(z, [self.config.batch_size, -1, self.output_dim])
            return z

    def Affine_Transformation(self, input_highway, output_dim, scope=None):
        ### input_highway : word embedding + char embedding - (batch * max_context, dim_sum)
        with tf.variable_scope(scope):
            w = tf.get_variable(name="highway_matrix", shape=[output_dim, input_highway.get_shape()[1]], dtype=tf.float32)
            b = tf.get_variable(name="highway_bias", shape=[output_dim], dtype=tf.float32)

            return tf.matmul(input_highway, w) + b

    def context_embedding(self, context_input, scope=None):
        ### utilizing contextual cues in word embedding
        ### captures interaction among context words independent of the query.

        with tf.variable_scope(scope):
            fw_cells = [tf.nn.rnn_cell.LSTMCell(self.output_dim, state_is_tuple=True) for _ in range(self.config.lstm_layers)]
            bw_cells = [tf.nn.rnn_cell.LSTMCell(self.output_dim, state_is_tuple=True) for _ in range(self.config.lstm_layers)]

            fw_dropout = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob) for cell in fw_cells]
            bw_dropout = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob) for cell in bw_cells]

            fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_dropout, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_dropout, state_is_tuple=True)

            output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, context_input, time_major=True, dtype=tf.float32)

            # forward output and backward output
            concat_output = tf.concat((output[0], output[1]), axis=2)

            return concat_output

    def similarity_matrix(self, cont_vec, ques_vec, scope=None):
        ### make similarity matrix using context vector and question vector
        ### cont_vec: (batch, context max, 2*dim)
        ### ques_vec: (batch, question max, 2*dim)

        assert cont_vec.get_shape()[2] == ques_vec.get_shape()[2]
        batch, cont, dim = cont_vec.get_shape()
        ques = ques_vec.get_shape()[1]

        with tf.variable_scope(scope):
            w = tf.get_variable("similarity_weight", [6*self.output_dim, 1], dtype=tf.float32) # 6d trainable weight vector

            # broadcasting the value of context and question vector
            ext_cont = tf.tile(tf.expand_dims(cont_vec, axis=2), [1, 1, ques, 1])
            ext_ques = tf.tile(tf.expand_dims(ques_vec, axis=1), [1, cont, 1, 1])

            dot_product = tf.multiply(ext_cont, ext_ques)
            dot_concat = tf.reshape(tf.concat((ext_cont, ext_ques, dot_product), axis=3), [-1, 6 * self.output_dim]) # [h;M;h*M]
            sim_mat = tf.reshape(tf.matmul(dot_concat, w), [self.config.batch_size, cont, ques])

            return sim_mat

    def context_to_query(self, sim_mat, ques_vec, scope=None):
        ### signifying which query words are most relevant to each context word
        with tf.variable_scope(scope):
            cont_attention = tf.nn.softmax(sim_mat, axis=2)
            cont2ques = tf.matmul(cont_attention, ques_vec)
            return cont2ques

    def query_to_context(self, sim_mat, cont_vec, scope=None):
        ### signifying which context words have the closest similarity to query word
        with tf.variable_scope(scope):
            ques_attention = tf.nn.softmax(tf.reduce_max(sim_mat, axis=2, keepdims=True)) # (32, 791, 1)
            ques2cont = tf.matmul(tf.reshape(ques_attention, [self.config.batch_size, 1, -1]), cont_vec) # (32, 1, 791) * (32, 791, 825 * 2)
            ques2cont = tf.squeeze(ques2cont, axis=1) # (32, 825 * 2)
            ques2cont = tf.tile(tf.expand_dims(ques2cont, axis=1), [1, sim_mat.get_shape()[1], 1]) # (32, 791, 825 * 2)
            return ques2cont

    def attention_flow(self, cont_vec, cont2ques, ques2cont, scope=None):
        ### encodese the query-aware representation of context words.

        with tf.variable_scope(scope):
            beta = tf.get_variable("attention_weight", [8*self.output_dim, 8*self.output_dim], dtype=tf.float32)
            cont_ques_dot = tf.multiply(cont_vec, cont2ques)
            cont_cont_dot = tf.multiply(cont_vec, ques2cont)
            attention_output = tf.concat((cont_vec, cont2ques, cont_ques_dot, cont_cont_dot), axis=2)
            attention_output = tf.matmul(tf.reshape(attention_output, [-1, attention_output.get_shape()[2]]), beta)
            attention_output = tf.reshape(attention_output, [self.config.batch_size, -1, 8*self.output_dim])

            return attention_output

    def modeling_layer(self, att_output, scope=None):
        ### captures the interaction among the context words conditioned on the query.
        with tf.variable_scope(scope):
            fw_cells = [tf.nn.rnn_cell.LSTMCell(self.output_dim, state_is_tuple=True) for _ in range(self.config.model_lstm_layers)]
            bw_cells = [tf.nn.rnn_cell.LSTMCell(self.output_dim, state_is_tuple=True) for _ in range(self.config.model_lstm_layers)]

            fw_dropout = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob) for cell in fw_cells]
            bw_dropout = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob) for cell in bw_cells]

            fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_dropout, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_dropout, state_is_tuple=True)

            output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, att_output, time_major=True, dtype=tf.float32)
            concat_output = tf.concat((output[0], output[1]), axis=2)

            return concat_output

    def output_layer(self, att_output, model_output, scope=None):
        with tf.variable_scope(scope):
            ### get start probability
            p1_w = tf.get_variable("start_output_weight", [10 * self.output_dim, 1], dtype=tf.float32)
            att_model_concat = tf.reshape(tf.concat((att_output, model_output), axis=2), [-1, 10*self.output_dim])
            self.att_model_linear = tf.reshape(tf.matmul(att_model_concat, p1_w), [self.config.batch_size, -1]) # (batch, 791)
            # p1 = tf.nn.softmax(att_model_linear, axis=1)
            p1 = tf.nn.softmax(self.att_model_linear)
            pred1 = tf.argmax(self.att_model_linear, axis=-1)

            ### get end probability
            p2_w = tf.get_variable("end_output_weight", [10*self.output_dim, 1], dtype=tf.float32)
            fw_cells = [tf.nn.rnn_cell.LSTMCell(self.output_dim, state_is_tuple=True) for _ in range(self.config.lstm_layers)]
            bw_cells = [tf.nn.rnn_cell.LSTMCell(self.output_dim, state_is_tuple=True) for _ in range(self.config.lstm_layers)]

            fw_dropout = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob) for cell in fw_cells]
            bw_dropout = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob) for cell in bw_cells]

            fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_dropout, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_dropout, state_is_tuple=True)

            output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, model_output, time_major=True, dtype=tf.float32)
            new_model_output = tf.concat((output[0], output[1]), axis=2)
            att_new_concat = tf.reshape(tf.concat((att_output, new_model_output), axis=2), [-1, 10*self.output_dim])
            self.att_new_linear = tf.reshape(tf.matmul(att_new_concat, p2_w), [self.config.batch_size, -1]) # (batch, 791)

            # p2 = tf.nn.softmax(att_new_linear, axis=1)
            p2 = tf.nn.softmax(self.att_new_linear)
            pred2 = tf.argmax(self.att_new_linear, axis=-1)

            return p1, p2, pred1, pred2

    def exp_moving_avg(self,):
        ### exponential moving average
        ### weighted moving average that gives more weighting to recent one rather than simple moving average
        pass

    def build_model(self, ):
        ### building total graph
        self.cont_word = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_cont])
        self.ques_word = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ques])
        self.cont_char = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_cont, self.config.max_cont_char])
        self.ques_char = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ques, self.config.max_ques_char])
        self.answer_start = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_cont])
        self.answer_stop  = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_cont])

        ### Word Embedding Layer
        cont_glove = self.word_embedding(self.cont_word, scope="context_word_emb")  # (32, 791, 300)
        ques_glove = self.word_embedding(self.ques_word, scope="question_word_emb") # (32, 60, 300)

        ### Character Embedding Layer
        cont_char = self.char_embedding(self.cont_char, is_query=False, scope="context_char_emb") # (32, 791, 525)
        ques_char = self.char_embedding(self.ques_char, is_query=True, scope="question_char_emb") # (32, 60, 525)

        ### make list to concat word embedding vector and character embedding vector
        cont_concat = tf.concat([cont_glove, cont_char], axis=2)  # (32, 791, 825)
        ques_concat = tf.concat([ques_glove, ques_char], axis=2)  # (32, 60, 825)
        self.output_dim = cont_concat.get_shape()[2] # (400)

        ### Highway network get dimension d
        cont_highway = self.highway(cont_concat, bias=-2.0, scope="context_highway_net")
        ques_highway = self.highway(ques_concat, bias=-2.0, scope="question_highway_net")

        ### Contextual Embedding Layer
        cont_output = self.context_embedding(cont_highway, scope="context_cont")   # (32, 791, 825 * 2)
        ques_output = self.context_embedding(ques_highway, scope="context_ques")   # (32, 60, 825 * 2)

        ### Attention Flow Layer
        sim_mat = self.similarity_matrix(cont_output, ques_output, scope="similarity_matrix")          # (32, 791, 60)
        cont2query = self.context_to_query(sim_mat, ques_output, scope="context2query")                # (32, 791, 825 * 2)
        query2cont = self.query_to_context(sim_mat, cont_output, scope="query2context")                # (32, 791, 825 * 2)
        att_output = self.attention_flow(cont_output, cont2query, query2cont, scope="attention_flow")  # (32, 791, 825 * 8)

        ### Modeling Layer
        model_output = self.modeling_layer(att_output, scope="modeling_layer") # (32, 791, 825 * 2)

        ### Output Layer
        prob1, prob2, self.pred1, self.pred2= self.output_layer(att_output, model_output, scope="output_layer")     # (32, 791), (32, 791)

        with tf.variable_scope("global_step"):
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
            self.ema = tf.train.ExponentialMovingAverage(self.config.decay_rate)

        with tf.name_scope("loss"):
            self.loss_p1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(self.answer_start, 'float'), \
                                                                        logits=self.att_model_linear)
            self.cross_entropy_p1 = tf.reduce_mean(self.loss_p1)
            self.loss_p2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(self.answer_stop, 'float'), \
                                                                        logits=self.att_new_linear)
            self.cross_entropy_p2 = tf.reduce_mean(self.loss_p2)

            self.loss = tf.add(self.loss_p1, self.loss_p2)
            self.cross_entropy = tf.add(self.cross_entropy_p1, self.cross_entropy_p2)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_opt = tf.train.AdadeltaOptimizer(self.config.lr).minimize(self.cross_entropy, global_step=self.global_step)

        # check the predicting index
        # start_idx = tf.argmax(self.answer_start, axis=-1)
        # stop_idx  = tf.argmax(self.answer_stop, axis=-1)

        # cor_pred1 = tf.equal(pred1, start_idx)
        # cor_pred2 = tf.equal(pred2, stop_idx)
        #
        # self.acc_pred1 = tf.reduce_mean(tf.cast(cor_pred1, tf.float32))
        # self.acc_pred2 = tf.reduce_mean(tf.cast(cor_pred2, tf.float32))

        # make f1 score
        # acc1 = tf.metrics.accuracy(labels=start_idx, predictions=pred1)
        # acc2 = tf.metrics.accuracy(labels=stop_idx, predictions=pred2)
        #
        # prec1 = tf.metrics.precision(labels=start_idx, predictions=pred1)
        # prec2 = tf.metrics.precision(labels=stop_idx, predictions=pred2)
        #
        # rec1 = tf.metrics.recall(labels=start_idx, predictions=pred1)
        # rec2 = tf.metrics.recall(labels=stop_idx, predictions=pred2)
        # pdb.set_trace()

    def model_summary(self, ):
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def init_saver(self, ):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

# for sanity check
def main():
    config = get_args()
    dataset = Squad_Dataset(config)
    model = BIDAF(config, dataset.word_idx2vec)

if __name__ == "__main__":
    main()

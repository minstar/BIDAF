import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('data_dir', './dataset/splitv2/', 'Multi-Sentence Reading Comprehension dataset direcitory')
flags.DEFINE_string('train_file', 'train_456-fixedIds', 'train filename')
flags.DEFINE_string('dev_file', 'dev_83-fixedIds', 'dev filename')
flags.DEFINE_string('glove_dir', './dataset/glove.840B.300d/', 'GLOVE 840B 300 dimension file path')
flags.DEFINE_string('glove_dir_6b', './dataset/glove.6B/', 'GLOVE 6B file path')
flags.DEFINE_string('glove_file', 'glove.840B.300d.txt', 'GLOVE 840B, 300d pretrained file name')
flags.DEFINE_string('glove_load', 'glove_dict.pkl', 'preprocessed GLOVE 840B, 300d pretrained file name')

flags.DEFINE_list('except1_idx', [52343, 151102, 209833, 220779], 'first key value error index list in glove')
flags.DEFINE_list('except2_idx', [128261, 200668, 253461, 365745, 532048, 717302, \
                                994818, 1123331, 1148409, 1352110, 1499727, 1533809, \
                                1899841, 1921152, 2058966, 2165246], 'second key value error index list in glove')
flags.DEFINE_list('kernel_features', [25, 50, 75, 100, 125, 150], 'kernel size as small data')
flags.DEFINE_list('kernel_width', [1,2,3,4,5,6], 'kernel width as small data')

flags.DEFINE_float('dropout', 0.5, 'LSTM dropout')

flags.DEFINE_integer('word_embedding_size', 300, 'glove word embedding sizes')
flags.DEFINE_integer('char_dimension', 20, 'choose the character dimension')
flags.DEFINE_integer('batch_size', 32, 'batch size of training time')
flags.DEFINE_integer('max_text', 615, 'max paragraph sizes')
flags.DEFINE_integer('max_char', 23, 'max length of one word')
flags.DEFINE_integer('max_num_query', 31, 'max number of query sizes')
flags.DEFINE_integer('max_query_text', 52, 'max query sizes')
flags.DEFINE_integer('max_query_char', 20, 'max length of one word in question')
flags.DEFINE_integer('max_num_answer', 21, 'max number of query answers')
flags.DEFINE_integer('max_answer', 90, 'max number of answer words')
flags.DEFINE_integer('LSTM_hidden', 500, 'number of LSTM hidden units') # 500
flags.DEFINE_integer('LSTM_layers', 2, 'number of LSTM layers')

# def main(_):
#     FLAGS = flags.FLAGS
#
#     m(config)
#
# if __name__ == "__main__":
#     tf.app.run()

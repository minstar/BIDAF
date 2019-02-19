import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('data_dir', './dataset/splitv2/', 'Multi-Sentence Reading Comprehension dataset direcitory')
flags.DEFINE_string('train_file', 'train_456-fixedIds', 'train filename')
flags.DEFINE_string('dev_file', 'dev_83-fixedIds', 'dev filename')
flags.DEFINE_string('glove_dir', './pretrained/glove.840B.300d/', 'GLOVE file path')
flags.DEFINE_string('glove_file', 'glove.840B.300d.txt', 'GLOVE 840B, 300d pretrained file name')
flags.DEFINE_string('glove_load', 'glove_dict.pkl', 'preprocessed GLOVE 840B, 300d pretrained file name')

flags.DEFINE_list('except1_idx', [52343, 151102, 209833, 220779], 'first key value error index list in glove')
flags.DEFINE_list('except2_idx', [128261, 200668, 253461, 365745, 532048, 717302, \
                                994818, 1123331, 1148409, 1352110, 1499727, 1533809, \
                                1899841, 1921152, 2058966, 2165246], 'second key value error index list in glove')

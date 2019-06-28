import argparse

def get_args():
    argp = argparse.ArgumentParser(description='BIDAF', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data direction
    argp.add_argument('--train_dir', type=str, default='./dataset/squad_v1.1/')
    argp.add_argument('--train_file', type=str, default='train-v1.1.json')
    argp.add_argument('--dev_file', type=str, default='dev-v1.1.json')
    argp.add_argument('--glove_dir', type=str, default='./dataset/glove.6B/')
    argp.add_argument('--glove_file', type=str, default='glove.6B.300d.txt')
    argp.add_argument('--glove_dict', type=str, default='glove_dict.pkl')

    # data control
    # argp.add_argument('--kernel_num', type=list, default=[100])
    # argp.add_argument('--kernel_width', type=list, default=[5])
    argp.add_argument('--max_ques', type=int, default=60)
    argp.add_argument('--max_ques_char', type=int, default=25)
    argp.add_argument('--max_cont', type=int, default=791)
    argp.add_argument('--max_cont_char', type=int, default=37)
    argp.add_argument('--batch_size', type=int, default=32)
    argp.add_argument('--char_dim', type=int, default=20)
    argp.add_argument('--char_6b', type=int, default=1259)
    argp.add_argument('--char_840b', type=int, default=0)

    # model control
    argp.add_argument('--filter_num', type=list, default=[25, 50, 75, 100, 125, 150])
    argp.add_argument('--filter_width', type=list, default=[1, 2, 3, 4, 5, 6])
    argp.add_argument('--highway_layers', type=int, default=2)
    argp.add_argument('--lstm_layers', type=int, default=1)
    argp.add_argument('--dropout', type=float, default=0.1)

    return argp.parse_args()

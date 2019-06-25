import argparse

def get_args():
    argp = argparse.ArgumentParser(description='BIDAF', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data direction
    argp.add_argument('--train_dir', type=str, default='./dataset/squad_v1.1/')
    argp.add_argument('--train_file', type=str, default='train-v1.1.json')
    argp.add_argument('--dev_file', type=str, default='dev-v1.1.json')
    argp.add_argument('--glove_dir', type=str, default='./pretrained/glove.840B.300d/')
    argp.add_argument('--glove_file', type=str, default='glove.840B.300d.txt')
    argp.add_argument('--glove_load', type=str, default='glove_dict.pkl')

    # data control
    argp.add_argument('--kernel_features', type=list, default=[100])
    argp.add_argument('--kernel_width', type=list, default=[5])
    argp.add_argument('--max_ques', type=int, default=60)
    argp.add_argument('--max_ques_char', type=int, default=25)
    argp.add_argument('--max_cont', type=int, default=791)
    argp.add_argument('--max_cont_char', type=int, default=37)
    argp.add_argument('--batch_size', type=int, default=32)

    # model control

    return argp.parse_args()

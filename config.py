import argparse

def get_args():
    argp = argparse.ArgumentParser(description='BIDAF', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data direction
    argp.add_argument('--train_dir', type=str, default='./dataset/squad_v1.1/')
    argp.add_argument('--train_file', type=str, default='train-v1.1.json')
    argp.add_argument('--dev_file', type=str, default='dev-v1.1.json')
    # argp.add_argument('--glove_dir', type=str, default='./dataset/glove.840B.300d/')
    # argp.add_argument('--glove_file', type=str, default='glove.840B.300d.txt')
    argp.add_argument('--glove_dir', type=str, default='./dataset/glove.6B/')
    argp.add_argument('--glove_file', type=str, default='glove.6B.300d.txt')
    argp.add_argument('--glove_dict', type=str, default='glove_dict.pkl')
    argp.add_argument('--save_dir', type=str, default='./out')
    argp.add_argument('--save_pred', type=str, default='./out/predictions.json')

    # data control
    argp.add_argument('--is_load', type=str, default='True')
    argp.add_argument('--max_ques', type=int, default=40)
    argp.add_argument('--max_ques_char', type=int, default=25)
    argp.add_argument('--max_cont', type=int, default=300) # original text is 791
    argp.add_argument('--max_cont_char', type=int, default=37)
    argp.add_argument('--batch_size', type=int, default=20)
    argp.add_argument('--char_dim', type=int, default=8)
    argp.add_argument('--char_6b', type=int, default=1311)
    # argp.add_argument('--max_cont_with_char', type=int, default=3054)

    # model control
    argp.add_argument('--filter_num', type=list, default=[100])
    argp.add_argument('--filter_width', type=list, default=[5])
    argp.add_argument('--highway_layers', type=int, default=2)
    argp.add_argument('--lstm_layers', type=int, default=1)
    argp.add_argument('--model_lstm_layers', type=int, default=2)
    argp.add_argument('--dropout', type=float, default=0.2)
    argp.add_argument('--lr', type=float, default=0.5)
    argp.add_argument('--decay_rate', type=float, default=0.999)
    argp.add_argument('--max_to_keep', type=int, default=20)

    # train control
    argp.add_argument('--epochs', type=int, default=1)
    argp.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'])
    argp.add_argument('--print_step', type=int, default=100)
    argp.add_argument('--loss_diff', type=float, default=0.1)
    return argp.parse_args()

import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    parser.add_argument('--path', type=str, default='data/ori_pqal.json', help='Path to data')
    parser.add_argument('--embedding_dim', type=int, default=300, help='(Token) Embedding size')
    parser.add_argument('--bs', type=int, default=512, help='Batch Size')

    parser.add_argument('--rnn_hidden_size', type=int, default=300, help='(Token) Embedding size')
    parser.add_argument('--rnn_layers', type=int, default=3, help='(Token) Embedding size')
    parser.add_argument('--epochs', type=int, default=100, help='Num epochs to train')
    parser.add_argument('--teacher_force_ratio', type=float, default=0.5, help='TF ratio')


    parser.add_argument('--bidirectional', type=bool, default=False, help='Path to data')
    parser.add_argument('--repeat_hid_state', type=bool, default=False, help='Path to data')

    # parser.add_argument('--embedding_dim', type=int, default=300, help='(Token) Embedding size')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the dataloader')
    parser.add_argument('--drop_last', type=bool, default=False, help='Drop the last batch')

    parser.add_argument('--run_name', type=str, default='test_out', help='Path to output')


    parser.add_argument('--use_pretrained_token_embeddings', type=bool, default=True, help='Self documenting')
    parser.add_argument('--pretrained_token_embeddings_path', type=str,
                        default="/scratch/gobi1/johnchen/wiki-news-300d-1M-subword.vec", help='Self documenting')

    # This will be automatically true if we are using pretrained_token_embeddings
    parser.add_argument('--use_tied_token_embeddings', type=bool, default=False, help='Self documenting')


    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and args.cuda
    args.device = torch.device("cuda:0" if use_cuda else "cpu")


    if not os.path.exists(args.run_name):
        os.mkdir(args.run_name)

    return args
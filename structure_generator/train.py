import warnings
warnings.filterwarnings("ignore")

import torch
import random
import argparse
import deepspeed

from solver import Solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='', help='identifier for directory')
    parser.add_argument('--output_dir', default='logs', help='base directory to save outputs')
    parser.add_argument('--data_root', default='data', help='root directory for data')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--device', type=str, default='cuda', help='')

    parser.add_argument('--channels', default=3, type=int)
    parser.add_argument('--ae_size', type=int, default=64, help='base dimensionality of hidden layer for auto-encoder')
    parser.add_argument('--rnn_size', type=int, default=256, help='base dimensionality of hidden layer for auto-regressive models')
    parser.add_argument('--K', type=int, default=1, help='multiplier on the dimensionality of hidden layer for frame auto-encoder')
    parser.add_argument('--M', type=int, default=1, help='multiplier on the dimensionality of hidden layer for frame auto-regressive models')
    parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of latent variables')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output and decoder input vectors')
    parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers of prior lstm model')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers posterior lstm model')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers of frame predicting lstm model')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--use_bn', action='store_true', default=False, help='whether to use batch normalization')

    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict during inference')

    parser.add_argument('--dataset', default='KITTI_64', help='dataset to train with')
    parser.add_argument('--data_threads', type=int, default=8, help='the number of data loading threads')
    parser.add_argument('--frame_sampling_rate', type=int, default=2, help='frame sampling rate')

    parser.add_argument('--max_iter', type=int, default=300, help='the number of iterations to train for')
    parser.add_argument('--print_iter', type=int, default=10, help='')
    parser.add_argument('--log_line_iter', type=int, default=50, help='')
    parser.add_argument('--log_img_iter', type=int, default=1000, help='')
    parser.add_argument('--log_ckpt_iter', type=int, default=1000, help='')
    parser.add_argument('--log_ckpt_sec', type=int, default=900, help='')
    parser.add_argument('--validate_iter', type=int, default=1, help='')

    parser.add_argument('--load_dp_ckpt', default=False, action='store_true', help='load model trained using data-parallel.')
    parser.add_argument('--load_ds_ckpt', default=False, action='store_true', help='load model trained using deepspeed.')

    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    opt = parser.parse_args()

    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    opt.inference = False
    solver = Solver(opt)
    solver.solve()

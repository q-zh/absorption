import os
import argparse
from solver import Solver
from data_loader import get_loader_test
from data_loader import get_loader_train
from data_loader import get_loader_val
from torch.backends import cudnn

def main(config):

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)


    cudnn.benchmark = True

    if config.mode == 'train' or config.mode == 'val':
        data_loader_train = get_loader_train(config)
        # data_loader_val = get_loader_val(config)
    else:
        data_loader_train = None
        # data_loader_val = None
  
    data_loader_test = get_loader_test(config)
    data_loader_val = None
    #data_loader_test = None
    solver = Solver(data_loader_train, data_loader_val, data_loader_test, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'val':
        solver.val_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.

    parser.add_argument('--main_dir', type=str, default='/home/qzheng/nips/dataset/')
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of total iterations for training D')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--n_critic', type=int, default=5)

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=59, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'val'])
    parser.add_argument('--use_tensorboard', type=bool, default=True)
    parser.add_argument('--num_epoch_decay', type=int, default=100)

    parser.add_argument('--distance_type', type=int, default=0, help='1: de-rain, 0: reflection removal')
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    # Directories.
    parser.add_argument('--log_dir', type=str, default='ours/logs')
    parser.add_argument('--model_save_dir', type=str, default='ours/models')
    parser.add_argument('--sample_dir', type=str, default='ours/samples')
    parser.add_argument('--result_dir', type=str, default='ours/results')

    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--lr_update_step', type=int, default=1000)
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)

    config = parser.parse_args()
    print(config)
    main(config)




import argparse

from Utils.memory_utils import BUDGETS


# Supported dataset names for experiments
DATASET_NAMES = ['caida', 'webdocs', 'mawi', 'retail', 'kosarak', 'synthetic'] 

# Distribution models for synthetic data generation
DISTRIBUTION_NAMES = ['real', 'zipf-icml', 'pareto', 'lognormal', 'exponential', 'zipf'] 

# Available decoding solvers
DECODING_SOLVERS = ['flore', 'lasso', 'omp', 'lsqr', 'lsmr', 'local'] 

# Memory budget options pulled from memory utilities
MEMORY_USAGES = BUDGETS.keys() 


def get_args(additional_args=[]):
    """
    Parse command-line arguments for the main experiment.
    
    This function defines and parses all command-line arguments used in the experiments,
    including data sources, model configurations, training hyperparameters, and testing options.
    
    Args:
        additional_args (list): List of additional arguments to add to the parser.
                               Each item should be a tuple of (name_or_flags, kwargs).
        
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Main Experiment')

    # experiment arguments
    parser.add_argument(
        '--seed', type=int, default=12345, 
        help='random seed')
    parser.add_argument(
        '--itrs', dest="num_exp_itrs", type=int, default=5, 
        help='experiments times')
    parser.add_argument(
        "--save-dname", dest="result_dir", type=str, default='./Results/', 
        help="where to save results")
    parser.add_argument(
        "--log-dname", dest="log_dname", type=str, default='./Logs/', 
        help="where to save logs")
    parser.add_argument(
        '--data-workers', dest='num_workers', type=int, default=0, 
        help='data loader num workers')
    parser.add_argument(
        '--is-testing', dest='test', default=False, action='store_true', 
        help='whether to test without training')
    parser.add_argument(
        '--is-measuring', dest='sim_time', default=False, action='store_true', 
        help='whether to compute processing time')

    # problem arguments
    parser.add_argument(
        "--data", type=str, default='caida', choices=DATASET_NAMES,
        help="data stream")
    parser.add_argument(
        "--stream-model", dest="distribution", type=str, default='real', choices=DISTRIBUTION_NAMES,
        help="stream model")
    parser.add_argument(
        "--data-fname", dest="data_fname", type=str, default='./Streams/caida.dat', 
        help=".dat file localization")  # need to be consistent with '--data' 
    parser.add_argument(
        '--samples', type=int, default=1_000_000,
        help='number of synthetic streaming items')
    
    # env hyper-parameters
    parser.add_argument(
        '--is-cuda', dest="is_cuda", default=False, action='store_true',
        help='whether to use cuda-version implementation') 
    parser.add_argument(
        '--interval', type=int, default=5000, 
        help='length of sampling inserval') 
    parser.add_argument(
        '--window', type=int, default=128, 
        help='length of maintained inservals') 
    parser.add_argument(
        '--break-point', dest="break_point", type=int, default=2_000_000, 
        help='finish simulation') 
    parser.add_argument(
        '--key-size', dest="key_size", type=int, default=13, 
        help='key size')
    parser.add_argument(
        '--budget', dest="mem_usage", type=str, default='1MB', choices=MEMORY_USAGES,
        help='memory budget: 16KB, 32KB, 64KB, ~, 1MB')
    parser.add_argument(
        '--stream-size', dest="stream_size", type=int, default=100_000, 
        help='expected stream size (number of distinct keys)') 
    
    # training hyper-parameters
    parser.add_argument(
        '--lr', dest='learning_rate', type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '--epochs', dest='num_epochs', type=int, default=25,
        help='number of training epochs')
    parser.add_argument(
        '--bsz', dest='batch_size', type=int, default=32,
        help='batch size')
    parser.add_argument(
        '--em-steps', dest='num_em_steps', type=int, default=3,
        help='number of EM steps')
    parser.add_argument(
        '--early-stop', default=False, action='store_true', 
        help='whether to stop early')  # Warning!!! Not Sopported Now!
    parser.add_argument(
        "--trade-offs", dest='weights', nargs="+", type=float, default=[1., 0.5, 0.1, 0., 0.01], 
        help="weights of different loss term: " \
             "1. consistency; 2. reconstruction; 3. orthogonal; 4. invertibility") 
    
    # model hyper-parameters
    parser.add_argument(
        '--layers', dest='num_layers', type=int, default=3,
        help='number of coupling layers')
    parser.add_argument(
        '--divide-len', dest='s_dim', type=int, default=1024, 
        help="length of decoding window")
    parser.add_argument(
        '--hidden-dim', dest='h_dim', type=int, default=64, 
        help="hidden dimmension of cINN")
    parser.add_argument(
        '--autoenc-x-dim', dest='ae_x_dim', type=int, default=256, 
        help="hidden dimmension of autoencoder-x")
    parser.add_argument(
        '--autoenc-y-dim', dest='ae_y_dim', type=int, default=192, 
        help="hidden dimmension of autoencoder-y")
    parser.add_argument(
        '--cond-dim', dest='c_dim', type=int, default=256, 
        help="max conditional dimmension of cINN")
    parser.add_argument(
        '--is-verbose', dest='is_verbose', default=False, action='store_true', 
        help='whether to print detailed information') 
    parser.add_argument(
        '--dropout', type=float, default=0., 
        help='dropout rate')
    parser.add_argument(
        "--ckpt", dest="ckpt_dir", type=str, default='./Weights/', 
        help="where to save model")
    parser.add_argument(
        '--print-model', dest='print_model', default=False, action='store_true', 
        help='whether to log or print model infomation')
    
    # testing hyper-parameters
    parser.add_argument(
        "--dec-mode", dest="decode_mode", type=str, choices=DECODING_SOLVERS, 
        help="decoding method")
    parser.add_argument(
        '--slice-test-start', dest="test_start", type=int, default=1,
        help="start interval index of testing")
    parser.add_argument(
        '--slice-test-stop', dest="test_stop", type=int, default=32,
        help="end interval index of testing")
    
    for add_arg in additional_args:
        name_or_flags, kwargs = add_arg[0], add_arg[1]
        parser.add_argument(name_or_flags, **kwargs)
    
    args = parser.parse_args()
    return args

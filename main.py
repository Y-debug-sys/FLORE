from helper import *

from Utils.log import *
from Utils.environment import *
from Utils.memory_utils import *
from Utils.useful_functions import *
from Utils.build import build_dataloader

from Baselines.classic_sketch.count import CountSketch
from Baselines.classic_sketch.count_min import CountMinSketch
from Baselines.compressed_sensing_sketch.pr_sketch import PRSketch
from Baselines.compressed_sensing_sketch.nze_sketch import SeqSketch
from Baselines.optimized_sketch.augmented import AugmentedSketch
from Baselines.optimized_sketch.conservative_update import ConservativeUpdateSketch

from FLORE.model import FLOREModel
from FLORE.solver import SketchingSolver
from Structure.sketch import FLORESketch


def main(args):
    """
    Main function to run the FLORE experiments.
    
    This function orchestrates the entire experimental process including:
    1. Setting up the environment and logging
    2. Configuring different sketch algorithms with appropriate memory budgets
    3. Running experiments for a specified number of iterations
    4. Training the FLORE model
    5. Evaluating results and collecting metrics
    6. Saving the final results
    
    Args:
        args: Command-line arguments containing experiment configuration
    """
    # Set random seed for reproducibility
    seed_everything(args.seed)

    # Initialize variables to store results
    all_results, name_seq = None, None
    # Get memory budget in bytes based on argument
    memory_bytes = BUDGETS[args.mem_usage]

    # Configure parameters for different sketch algorithms based on memory budget
    cms_param_dict = cm_or_cs_config(memory_bytes)
    pr_param_dict = pr_config(memory_bytes, stream_size=args.stream_size)
    nze_param_dict = nze_config(memory_bytes, args.key_size, stream_size=args.stream_size)
    # ls_param_dict = ls_config(memory_bytes, args.key_size, has_model=True)
    flore_param_dict = flore_config(memory_bytes, args.key_size, stream_size=args.stream_size)
    ag_param_dict = ag_config(memory_bytes, args.key_size)

    if args.data == 'synthetic':
        data_name = args.distribution
    else:
        data_name = args.data

    # Setup directories for checkpoints, results and logs
    ckpt_dir = args.ckpt_dir + f'{data_name}/'
    result_dir = args.result_dir + f'{data_name}/'
    log_dir = args.log_dname + f'{data_name}/'
    check_and_create_dir(ckpt_dir)
    check_and_create_dir(result_dir)
    check_and_create_dir(log_dir)

    # Setup logger
    logger = setup_logger(log_dir + f'log_{args.mem_usage}.log')
    logger.info("Starting new experiment with the following key args: " \
                "memory usage = {}; repeated times = {}; dataset name = {}.".format(args.mem_usage, args.num_exp_itrs, data_name))
    logger.info("Sketch parameters are as follows:\n"\
                "-- Count-Min Sketch (cm), Count Sketch (cs) and Conservative-Update Sketch (cu): {}\n"\
                "-- PR-sketch (pr): {}\n"\
                "-- NZE-sketch (nze): {}\n"\
                # "-- Learning-Augmented Sketches (lcm, lcs, ls): {}\n"\
                "-- Augmented Sketch (ag): {}\n"\
                "-- FLORE (flore): {}\n".format(
                    ", ".join(f"{k}={v}" for k, v in sorted(cms_param_dict.items())), 
                    ", ".join(f"{k}={v}" for k, v in sorted(pr_param_dict.items())), 
                    ", ".join(f"{k}={v}" for k, v in sorted(nze_param_dict.items())), 
                    # ", ".join(f"{k}={v}" for k, v in sorted(ls_param_dict.items())), 
                    ", ".join(f"{k}={v}" for k, v in sorted(ag_param_dict.items())), 
                    ", ".join(f"{k}={v}" for k, v in sorted(flore_param_dict.items()))
                ))

    # Run experiments for specified number of iterations
    for ii in range(args.num_exp_itrs):
        logger.info(f"Experimental iteration {ii+1} of {args.num_exp_itrs} starting ...")
        sketches = {}

        # Initialize classic sketch algorithms
        cs = CountSketch(**cms_param_dict, KEY_T_SIZE=args.key_size)
        cm = CountMinSketch(**cms_param_dict, KEY_T_SIZE=args.key_size)
        sketches['cs'], sketches['cm'] = cs, cm

        # Initialize optimized sketch algorithms
        cu = ConservativeUpdateSketch(**cms_param_dict, KEY_T_SIZE=args.key_size)
        ag = AugmentedSketch(**ag_param_dict, KEY_T_SIZE=args.key_size)
        sketches['cu'], sketches['ag'] = cu, ag

        # Initialize compressed sensing sketch algorithms
        pr = PRSketch(**pr_param_dict, KEY_T_SIZE=args.key_size)        
        nze = SeqSketch(**nze_param_dict, KEY_T_SIZE=args.key_size)
        sketches['pr'], sketches['nze'] = pr, nze

        # # Initialize learning-augmented sketch algorithms
        # lcm = LearnedSketch(**ls_param_dict, sketch='cm', KEY_T_SIZE=args.key_size)
        # lcs = LearnedSketch(**ls_param_dict, sketch='cs', KEY_T_SIZE=args.key_size)
        # ls = LearnedSketchW(**ls_param_dict, KEY_T_SIZE=args.key_size)
        # ls.set_nkey(args.stream_size, factor=10.0)
        # sketches['lcm'], sketches['lcs'], sketches['ls'] = lcm, lcs, ls

        # Initialize FLORE sketch algorithm
        flore = FLORESketch(**flore_param_dict, KEY_T_SIZE=args.key_size, decoding_method='flore')
        sketches['flore'] = flore

        # Select appropriate environment based on data source and distribution
        if data_name in ['caida', 'wawi']:
            sim_env = TraceEnv(args, logger, sketches)
        elif args.distribution == 'real':
            sim_env = DataEnv(args, logger, sketches)
        else:
            sim_env = GenEnv(args, logger, sketches)

        # Run simulation environment
        sim_env.run_fast(break_point=args.break_point) 

        # Get problem dimensions and sketching matrix for FLORE model
        num_ydim, num_xdim = sim_env.sketches['flore'].get_problem_MN()

        if num_xdim == 0:
            flore_solver = None
            sim_env.sketches['flore'].decoding_method = 'local'
        else:
            # Initialize FLORE model and solver
            gen_model = FLOREModel(args.s_dim, args.ae_x_dim, num_ydim, args.ae_y_dim, args.dropout, num_layers=args.num_layers, 
                                   hidden_dim=args.h_dim, max_cond_dim=args.c_dim, verbose=args.is_verbose)
            # print(gen_model)
            flore_solver = SketchingSolver(args, gen_model, ckpt_dir, logger=logger, 
                                           loss_type=[SketchingSolver.MMD, SketchingSolver.MSE], 
                                           alphas=args.weights) 
        
            # Get training data and train (or load) FLORE model
            snapshots, sketching_matrix = sim_env.get_samples_in_window(), sim_env.sketches['flore'].get_sketching_matrix(True)
            train_loader, *_ = build_dataloader(args, snapshots, sketching_matrix)
            flore_solver.update_sys(num_xdim, sketching_matrix)

            info='{}_{}'.format(args.mem_usage, ii)
            if args.test:
                flore_solver.load(filename=f"model_{info}.pt")
            else:
                flore_solver.train(train_loader, info, num_epochs=args.num_epochs)

        # Update learned sketch parameters and recover results
        all_results_ii, name_seq_ii = sim_env.recover(nn_solver=flore_solver, is_clear=True)
        _, *_ = get_evaluation_metrics(all_results_ii, name_seq_ii, logger=logger)
        
        # Accumulate results across iterations
        if all_results is None: 
            all_results, name_seq = np.expand_dims(all_results_ii, axis=0), name_seq_ii
        else:
            all_results_ii = np.expand_dims(all_results_ii, axis=0)
            all_results, name_seq = merge_results_by_name(all_results, name_seq, all_results_ii, name_seq_ii)
        
        if args.sim_time:
            # Measure processing time and log performance metrics
            time_dict = sim_env.get_processing_time(100_000, is_random=True)
            # lcm.set_mode(run_oracle=True), lcs.set_mode(run_oracle=True), ls.set_mode(run_oracle=True)
            logger.info("Processing cost are as follows:\n"\
                    "-- cm: total space = {:.2f}KB, per time = {:.2f}us || cs: total space = {:.2f}KB, per time = {:.2f}us \n"\
                    "-- cu: total space = {:.2f}KB, per time = {:.2f}us || ag: total space = {:.2f}KB, per time = {:.2f}us \n"\
                    "-- pr: total space = {:.2f}KB, per time = {:.2f}us || nze: total space = {:.2f}KB, per time = {:.2f}us \n"\
                    # "-- lcm: total space = {:.2f}KB, per time = {:.2f}us || lcs: total space = {:.2f}KB, per time = {:.2f}us || ls: total space = {:.2f}KB, per time = {:.2f}us \n"\
                    "-- FLORE (flore): total space = {:.2f}KB, per time = {:.2f}us \n".format(
                        cm.get_memory_usage() / 1024, time_dict['cm'] * 10**6, cs.get_memory_usage() / 1024, time_dict['cs'] * 10**6, 
                        cu.get_memory_usage() / 1024, time_dict['cu'] * 10**6, ag.get_memory_usage() / 1024, time_dict['ag'] * 10**6, 
                        pr.get_memory_usage() / 1024, time_dict['pr'] * 10**6, nze.get_memory_usage() / 1024, time_dict['nze'] * 10**6, 
                        # lcm.get_memory_usage() / 1024, time_dict['lcm'] * 10**6, lcs.get_memory_usage() / 1024, time_dict['lcs'] * 10**6, 
                        # ls.get_memory_usage() / 1024, time_dict['ls'] * 10**6, 
                        flore.get_memory_usage() / 1024, time_dict['flore'] * 10**6
                    ))

        logger.info(f"Experimental iteration {ii+1} of {args.num_exp_itrs} finished ...\n")
        
    # Save final results to pickle file
    fname = result_dir + f'results_{args.mem_usage}.pkl'
    save_to_pickle(all_results, name_seq, data_path=fname)


if __name__ == "__main__":
    # Parse command line arguments and run main function
    args = get_args()
    main(args)
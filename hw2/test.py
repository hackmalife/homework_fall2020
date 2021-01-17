import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v0")
parser.add_argument('--exp_name', type=str, default='todo')
parser.add_argument('--n_iter', '-n', type=int, default=200)

parser.add_argument('--reward_to_go', '-rtg', action='store_true')
parser.add_argument('--nn_baseline', action='store_true')
parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration
parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration

parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
parser.add_argument('--discount', type=float, default=1.0)
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
parser.add_argument('--n_layers', '-l', type=int, default=2)
parser.add_argument('--size', '-s', type=int, default=64)

parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--no_gpu', '-ngpu', action='store_true')
parser.add_argument('--which_gpu', '-gpu_id', default=0)
parser.add_argument('--video_log_freq', type=int, default=-1)
parser.add_argument('--scalar_log_freq', type=int, default=1)

parser.add_argument('--save_params', action='store_true')

args, unknown = parser.parse_known_args()

params = vars(args)

params['env_name'] = 'HalfCheetah-v2'
params['ep_len'] = 150
params['discount'] = 0.95
params['n_iter'] = 100
params['size'] = 32
# params['reward_to_go'] = True
params['nn_baseline'] = True

# params['dont_standardize_advantages'] = True
# params['exp_name'] = 'q1_lb_rtg_na'


##################################
### CREATE DIRECTORY FOR LOGGING
##################################

data_path = os.path.join(os.getcwd(), 'data')

if not (os.path.exists(data_path)):
    os.makedirs(data_path)

import time

# search hyperparameter
bsizes = [ 30000 ]
# lrs = [1e-3, 5e-3, 1e-2, 5e-2]
# bsizes = [ 1000 ]
lrs = [ 2e-2 ]

for bsize in bsizes:
    params['batch_size'] = bsize
    ## ensure compatibility with hw1 code
    params['train_batch_size'] = params['batch_size']
    for lr in lrs:
        params['learning_rate'] = lr
        # params['exp_name'] = 'q2_b'+ str(bsize) +'_r'+str(lr)
        # params['exp_name'] = 'q3_b40000_r0.005'
        # params['exp_name'] = 'q4_search_b' + str(bsize) + '_lr' + str(lr) + '_rtg_nnbaseline'
        params['exp_name'] = 'q4_b' + str(bsize) + '_r' + str(lr) + '_nnbaseline'
        params['video_log_freq'] = 100
        # params['exp_name'] = 'q1_sb_no_rtg_dsa'

        logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(data_path, logdir)
        params['logdir'] = logdir
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)

        from cs285.scripts.run_hw2 import PG_Trainer

        trainer = PG_Trainer(params)
        trainer.run_training_loop()

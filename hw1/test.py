import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--expert_policy_file', '-epf', type=str,
                    default='cs285/policies/experts/Ant.pkl')  # relative to where you're running this script from
parser.add_argument('--expert_data', '-ed', type=str,
                    default='cs285/expert_data/expert_data_Ant-v2.pkl') # relative to where you're running this script from
parser.add_argument('--env_name', '-env', type=str,
                    help='choices: Ant-v2, Humanoid-v2, Walker-v2, HalfCheetah-v2, Hopper-v2',
                    default='Ant-v2')
parser.add_argument('--exp_name', '-exp', type=str,
                    help='pick an experiment name', default='dagger_ant')
parser.add_argument('--do_dagger', action='store_true')
parser.add_argument('--ep_len', type=int)

parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
parser.add_argument('--n_iter', '-n', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
parser.add_argument('--eval_batch_size', type=int,
                    default=5000)  # eval data collected (in the env) for logging metrics
parser.add_argument('--train_batch_size', type=int,
                    default=100)  # number of sampled data points to be used per gradient/train step

parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

parser.add_argument('--video_log_freq', type=int, default=5)
parser.add_argument('--scalar_log_freq', type=int, default=1)
parser.add_argument('--no_gpu', '-ngpu', action='store_true')
parser.add_argument('--which_gpu', type=int, default=0)
parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
parser.add_argument('--save_params', action='store_true')
parser.add_argument('--seed', type=int, default=1)
# args = parser.parse_args()
args, unknown = parser.parse_known_args()

# convert args to dictionary
params = vars(args)

# for debugging
# params['no_gpu'] = True

# for DAgger
params['eval_batch_size'] = 5000
params['do_dagger'] = True
params['exp_name'] = 'dagger_ant'
params['n_iter'] = 10

# for Humanoid-v2
params['expert_policy_file'] = 'cs285/policies/experts/Humanoid.pkl'
params['expert_data'] = 'cs285/expert_data/expert_data_Humanoid-v2.pkl'
params['env_name'] = 'Humanoid-v2'
params['exp_name'] = 'dagger_humanoid'

# for HalfCheetah-v2
params['expert_policy_file'] = 'cs285/policies/experts/HalfCheetah.pkl'
params['expert_data'] = 'cs285/expert_data/expert_data_HalfCheetah-v2.pkl'
params['env_name'] = 'HalfCheetah-v2'
params['exp_name'] = 'dagger_cheetah'

# for Walker2d-v2
params['expert_policy_file'] = 'cs285/policies/experts/Walker2d.pkl'
params['expert_data'] = 'cs285/expert_data/expert_data_Walker2d-v2.pkl'
params['env_name'] = 'Walker2d-v2'
params['exp_name'] = 'dagger_walker2d'

# for Hopper-v2
params['expert_policy_file'] = 'cs285/policies/experts/Hopper.pkl'
params['expert_data'] = 'cs285/expert_data/expert_data_Hopper-v2.pkl'
params['env_name'] = 'Hopper-v2'
params['exp_name'] = 'dagger_hopper'


import os
import time

logdir_prefix = 'q1_'

data_path = os.path.join(os.getcwd(), 'data')
if not (os.path.exists(data_path)):
    os.makedirs(data_path)
logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
logdir = os.path.join(data_path, logdir)
params['logdir'] = logdir
if not(os.path.exists(logdir)):
    os.makedirs(logdir)

from cs285.scripts.run_hw1 import BC_Trainer

trainer = BC_Trainer(params)
trainer.run_training_loop()

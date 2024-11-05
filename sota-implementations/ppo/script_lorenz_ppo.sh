#!/bin/bash
#PBS -lwalltime=15:00:00
#PBS -l select=1:ncpus=10:mem=12gb:ngpus=1

cd $PBS_O_WORKDIR

# Cluster Environment Setup
# module load anaconda3/personal
source ~/anaconda3/bin/activate
source activate ppo2

# export LD_LIBRARY_PATH=~/anaconda3/envs/dreamer_flows/lib/:$LD_LIBRARY_PATH

echo "Launched program!"

export WANDB_API_KEY=4cd148f8d67db0bd8a73083368f31dcdd759869c

# !(python ppo_beacon.py env.env_name=lorenz collector.max_episode_length=400 \
#   logger.project_name=Lorenz_PPO_test \
#   logger.exp_name=Lorenz_PPO_test)

python ppo_beacon.py env.env_name=lorenz \
       collector.max_episode_length=400 logger.test_interval=200000 env.num_envs=5 \
       logger.project_name=Lorenz_PPO_test \
       logger.exp_name=Lorenz_PPO_test_numenvs5_3

echo "finished training!"

exit 0

from torchrl.envs import GymWrapper
import torch.nn, torch
import torch.optim
import numpy as np
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    InitTracker,
    FiniteTensorDictCheck,
    TransformedEnv,
    VecNorm,
    ParallelEnv,
    Compose,
    ObservationNorm,
    EnvCreator,
)
# from utils.rng import env_seed
import importlib, sys
module_path = '/rds/general/user/pg221/home/PhD_projects2/basic_sac_ks/beacon/lorenz'
sys.path.append(module_path)
from beacon.lorenz import lorenz
from torchrl.data import OneHotDiscreteTensorSpec


def add_env_transforms(env, cfg):
    transform_list = [
        InitTracker(),
        RewardSum(),
        StepCounter(cfg.collector.max_episode_length // cfg.env.frame_skip),
        FiniteTensorDictCheck(),
    ]

    transforms = Compose(*transform_list)
    return TransformedEnv(env, transforms)

def make_beacon_env(name, cfg):
    module = importlib.import_module(f"beacon.{name}")
    obj = getattr(module, name)
    gym_env = obj()
    gym_env = GymWrapper(gym_env)
    new_shape = torch.ones(gym_env.action_spec.shape)[None].size()
    print(gym_env.action_spec)
    print(gym_env.reset())
    # gym_env.action_spec = OneHotDiscreteTensorSpec(
    #                                 n = new_shape[-1],
    #                                 shape  = new_shape,
    #                                 device = gym_env.action_spec.device,
    #                                 dtype  = gym_env.action_spec.dtype,
    #                       )
    # print(gym_env.action_spec)

        # Create environments
    # gym_env = add_env_transforms(gym_env, cfg)
    # train_env.set_seed(env_seed(cfg))
    return gym_env

def make_parallel_beacon_env(name, cfg):
    make_env_fn = EnvCreator(lambda: make_beacon_env(name, cfg))
    env = ParallelEnv(cfg.env.num_envs, make_env_fn)
    env.reset()
    # print("in make_parallel: ", env.action_spec)
    return env

if __name__ =='__main__':
    name = 'lorenz'
    # torchrl_env = make_parallel_beacon_env(name)
    # print(torchrl_env)
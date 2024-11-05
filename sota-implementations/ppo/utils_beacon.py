# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from torchrl.data import Composite
from torchrl.data.tensor_specs import CategoricalBox
from torchrl.envs import GymWrapper
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvCreator,
    ExplorationType,
    GrayScale,
    GymEnv,
    NoopResetEnv,
    ParallelEnv,
    RenameTransform,
    Resize,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
# from utils.rng import env_seed
import importlib, sys
module_path = '/rds/general/user/pg221/home/PhD_projects2/rl/sota-implementations/ppo/beacon/lorenz'
sys.path.append(module_path)
module_path = '/rds/general/user/pg221/home/PhD_projects2/rl/sota-implementations/ppo/beacon/mixing'
sys.path.append(module_path)
from beacon.lorenz import lorenz
from beacon.mixing import mixing
from torchrl.record import VideoRecorder


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_base_env(env_name="lorenz", is_test = False):
    module = importlib.import_module(f"beacon.{env_name}")
    obj = getattr(module, env_name)
    env = obj()
    env = GymWrapper(env)
    env = TransformedEnv(env)
    env.append_transform(NoopResetEnv(noops=30, random=True))
    # print("obs spec: ", env.observation_spec)
    # if not is_test:
    #     env.append_transform(EndOfLifeTransform())
    return env


def make_parallel_env(env_name, num_envs, cfg, device, is_test=False):
    env = ParallelEnv(
        num_envs,
        EnvCreator(lambda: make_base_env(env_name)),
        serial_for_single=True,
        device=device,
    )
    env = TransformedEnv(env)
    # env.append_transform(RenameTransform(in_keys=["pixels"], out_keys=["pixels_int"]))
    # env.append_transform(ToTensorImage(in_keys=["pixels_int"], out_keys=["pixels"]))
    # env.append_transform(GrayScale())
    # env.append_transform(Resize(84, 84))
    # env.append_transform(CatFrames(N=4, dim=-3))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=cfg.collector.max_episode_length // cfg.collector.frame_skip))
    # if not is_test:
    #     env.append_transform(SignTransform(in_keys=["reward"]))
    env.append_transform(DoubleToFloat())
    env.append_transform(VecNorm(in_keys=["observation"]))
    return env

# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_modules_pixels(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape[-1]
    print("input_shape: ", input_shape)
    # Define distribution class and kwargs
    if isinstance(proof_environment.action_spec.space, CategoricalBox):
        num_outputs = proof_environment.action_spec.space.n
        distribution_class = OneHotCategorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = proof_environment.action_spec.shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": proof_environment.action_spec.space.low,
            "high": proof_environment.action_spec.space.high,
        }

    # Define input keys
    in_keys = ["observation"]

    # Define a shared Module and TensorDictModule (CNN + MLP)
    # common_cnn = ConvNet(
    #     activation_class=torch.nn.ReLU,
    #     num_cells=[32, 64, 64],
    #     kernel_sizes=[8, 4, 3],
    #     strides=[4, 2, 1],
    # )
    # common_cnn_output = common_cnn(torch.ones(input_shape))
    common_mlp = MLP(
        in_features=input_shape,
        activation_class=torch.nn.ReLU,
        activate_last_layer=True,
        out_features=512,
        num_cells=[256, 256],
    )
    # common_mlp_output = common_mlp(common_cnn_output)
    common_mlp_output = common_mlp(torch.ones(input_shape))


    # Define shared net as TensorDictModule
    common_module = TensorDictModule(
        module=torch.nn.Sequential(common_mlp),
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # Define on head for the policy
    policy_net = MLP(
        in_features=common_mlp_output.shape[-1],
        out_features=num_outputs,
        activation_class=torch.nn.ReLU,
        num_cells=[256, 256],
    )
    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=Composite(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define another head for the value
    value_net = MLP(
        activation_class=torch.nn.ReLU,
        in_features=common_mlp_output.shape[-1],
        out_features=1,
        num_cells=[256, 256],
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module


def make_ppo_models(env_name, cfg):

    proof_environment = make_parallel_env(env_name, 1, cfg, device="cpu")
    common_module, policy_module, value_module = make_ppo_modules_pixels(
        proof_environment
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    )

    with torch.no_grad():
        td = proof_environment.rollout(max_steps=cfg.collector.max_episode_length // cfg.collector.frame_skip, break_when_any_done=False)
        td = actor_critic(td)
        del td

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    del proof_environment

    return actor, critic


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()


def eval_model(actor, test_env, cfg):# max_steps = 1000, num_episodes=3):
    test_rewards = []
    num_episodes = cfg.logger.num_test_episodes
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=cfg.collector.max_episode_length // cfg.collector.frame_skip,
        )
        test_env.apply(dump_video)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
    del td_test
    return torch.cat(test_rewards, 0).mean()


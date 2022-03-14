#!/usr/bin/python3

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
import pdb
from signal import pause
from numpy.core.overrides import ARRAY_FUNCTION_ENABLED
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import pickle as pkl
import sys
from stable_baselines.common.callbacks import CheckpointCallback
from motion_imitation.learning import ppo_imitation as ppo_imitation
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.envs import env_builder as env_builder
import time
import tensorflow as tf
import random
import numpy as np
from mpi4py import MPI
import argparse
import time as tm



TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True


def set_rand_seed(seed=None):
    if seed is None:
        seed = int(time.time())

    seed += 97 * MPI.COMM_WORLD.Get_rank()

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return


def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
    policy_kwargs = {
        "net_arch": [{"pi": [512, 256],
                      "vf": [512, 256]}],
        "act_fun": tf.nn.relu
    }

    timesteps_per_actorbatch = int(
        np.ceil(float(timesteps_per_actorbatch) / num_procs))
    optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

    model = ppo_imitation.PPOImitation(
        policy=imitation_policies.ImitationPolicy,
        env=env,
        gamma=0.95,
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=0.2,
        optim_epochs=1,
        optim_stepsize=1e-5,
        optim_batchsize=optim_batchsize,
        lam=0.95,
        adam_epsilon=1e-5,
        schedule='constant',
        policy_kwargs=policy_kwargs,
        tensorboard_log=output_dir,
        verbose=1)
    return model


def train(model, env, total_timesteps, output_dir="", int_save_freq=0):
    if (output_dir == ""):
        save_path = None
    else:
        save_path = os.path.join(output_dir, "model.zip")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    callbacks = []
    # Save a checkpoint every n steps
    if (output_dir != ""):
        if (int_save_freq > 0):
            int_dir = os.path.join(output_dir, "intermedate")
            callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                                name_prefix='model'))

    model.learn(total_timesteps=total_timesteps,
                save_path=save_path, callback=callbacks)

    return


def test(model, env, num_procs, num_episodes=None):
    curr_return = 0
    sum_return = 0
    episode_count = 0

    if num_episodes is not None:
        num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
    else:
        num_local_episodes = np.inf

    o = env.reset()
    while episode_count < num_local_episodes:
        a, _ = model.predict(o, deterministic=True)
        o, r, done, info = env.step(a)
        curr_return += r
        if done:
            o = env.reset()
            sum_return += curr_return
            episode_count += 1

    sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
    episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

    mean_return = sum_return / episode_count

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Mean Return: " + str(mean_return))
        print("Episode Count: " + str(episode_count))

    return


def rollout(model, env):
    """ Roll out with the model simulated in env

    Args: 
        model: trained control policy
        env: locomotion_gyn_env

    Returns:
        o_arr: An array of observations in numpy array
        a_arr: An array of actions in numpy array
    """
    curr_return = 0
    sum_return = 0
    episode_count = 0

    num_local_episodes = 1

    o = env.reset()
    o_arr = np.array([])
    a_arr = np.array([])
    torque_arr = np.array([])
    ctacts_arr = np.array([])
    pos_arr, rpy_arr = np.array([]), np.array([])
    vel_arr, rpyrate_arr = np.array([]), np.array([])
    q_arr, qd_arr = np.array([]), np.array([])
    time_arr = np.array([])
    time = 0
    time_step = 0.033

    while episode_count < num_local_episodes:
        # get contact status at current step
        ctacts = env.GetFootContacts()
        # get base information at current step
        pos, rpy, vel, rpyrate = env.GetTrueBaseInformation()
        # get (actuated) joint information at current step
        q, qd = env.GetTrueJointInformation()
        # get action from the policy for the current step
        a, _ = model.predict(o, deterministic=True)
        # perform one-step forward simulation using PD law
        o, r, done, info = env.step(a)
        # get the applied torques for the current step
        torque = env.GetAppliedMotorTorques()

        # Convert to numpy array
        pos, rpy, vel, rpyrate = np.asarray(pos), np.asarray(
            rpy), np.asarray(vel), np.asarray(rpyrate)
        ctacts = np.asarray(ctacts)
        q, qd = np.asarray(q), np.asarray(qd)

        # Stack into trajectories
        o_arr = np.vstack((o_arr, o)) if o_arr.size else o
        a_arr = np.vstack((a_arr, a)) if a_arr.size else a
        torque_arr = np.vstack((torque_arr, torque)
                               ) if torque_arr.size else torque
        ctacts_arr = np.vstack((ctacts_arr, ctacts)
                               ) if ctacts_arr.size else ctacts
        pos_arr = np.vstack((pos_arr, pos)) if pos_arr.size else pos
        rpy_arr = np.vstack((rpy_arr, rpy)) if rpy_arr.size else rpy
        vel_arr = np.vstack((vel_arr, vel)) if vel_arr.size else vel
        rpyrate_arr = np.vstack((rpyrate_arr, rpyrate)
                                ) if rpyrate_arr.size else rpyrate
        q_arr = np.vstack((q_arr, q)) if q_arr.size else q
        qd_arr = np.vstack((qd_arr, qd)) if qd_arr.size else qd
        time_arr = np.append(time_arr, time)
        curr_return += r
        time += time_step
        # tm.sleep(1)        
        if done:
            o = env.reset()
            sum_return += curr_return
            episode_count += 1

    sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
    episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

    mean_return = sum_return / episode_count

    state_traj = (rpy_arr, pos_arr, rpyrate_arr, vel_arr, q_arr, qd_arr)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Mean Return: " + str(mean_return))
        print("Episode Count: " + str(episode_count))

    data = (time_arr, o_arr, a_arr, torque_arr, ctacts_arr, state_traj)
    fname = currentdir + '/data/rollout/traj_data.pickle'

    with open(fname, 'wb') as f:
        pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

    return




def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--robot", dest="robot",
                            type=str, default="Laikago")
    arg_parser.add_argument("--motion_file", dest="motion_file",
                            type=str, default="motion_imitation/data/motions/dog_pace.txt")
    arg_parser.add_argument("--visualize", dest="visualize",
                            action="store_true", default=False)
    arg_parser.add_argument(
        "--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--num_test_episodes",
                            dest="num_test_episodes", type=int, default=None)
    arg_parser.add_argument(
        "--model_file", dest="model_file", type=str, default="")
    arg_parser.add_argument("--total_timesteps",
                            dest="total_timesteps", type=int, default=2e8)
    # save intermediate model every n policy steps
    arg_parser.add_argument(
        "--int_save_freq", dest="int_save_freq", type=int, default=0)

    args = arg_parser.parse_args()
    robot_list = ["Laikago", "A1", "MC"]

    if args.robot not in robot_list:
        print("robot does not exist")
        return

    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    mode = "test" if args.mode == "rollout" else args.mode

    enable_env_rand = ENABLE_ENV_RANDOMIZER and (mode != "test")

    env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                          num_parallel_envs=num_procs,
                                          mode=mode,
                                          enable_randomizer=enable_env_rand,
                                          enable_rendering=args.visualize,
                                          plane = True, robot_type=args.robot)

    model = build_model(env=env,
                        num_procs=num_procs,
                        timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
                        optim_batchsize=OPTIM_BATCHSIZE,
                        output_dir=args.output_dir)

    if args.model_file != "":
        model.load_parameters(args.model_file)

    if args.mode == "train":
        train(model=model,
              env=env,
              total_timesteps=args.total_timesteps,
              output_dir=args.output_dir,
              int_save_freq=args.int_save_freq)
    elif args.mode == "test":
        test(model=model,
             env=env,
             num_procs=num_procs,
             num_episodes=args.num_test_episodes)
    elif args.mode == "rollout":
        rollout(model=model, env=env)
    else:
        assert False, "Unsupported mode: " + args.mode

    return


if __name__ == '__main__':
    main()

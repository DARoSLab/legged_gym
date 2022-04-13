# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.experiment.recompute_normalization = False
    print("sim_dt: ", env_cfg.sim.dt)
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        RS_PATH = '/home/dan/DARo/config/'
        mean = env.obs_buf_mean_cummulative.cpu().numpy().reshape(-1)
        var = env.obs_buf_var_cummulative.cpu().numpy().reshape(-1)
        np.savetxt("/home/dan/DARo/config/mean.csv", mean, delimiter=",")
        np.savetxt("/home/dan/DARo/config/var.csv", var, delimiter=",")

        export_policy_as_jit(ppo_runner.alg.actor_critic, RS_PATH, model_name='{}.pt'.format(args.task))
        print('#'*100)
        print('exported to {}'.format(RS_PATH))
        print('#'*100)

        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 1 # which robot is used for logging
    joint_index = 0 # which joint is used for logging
    leg_index = 0 #which leg is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1., 1., 0.])
    camera_vel = np.array([1., 1., 1.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    overide_rand_command = False
    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        # print_obs(obs)
        # if(i>=2):
        #     quit()
        if overide_rand_command:
            for r_idx in range(env_cfg.env.num_envs):
                env.commands[r_idx, 0] = 0.5
                env.commands[r_idx, 1] = 0.0
                env.commands[r_idx, 2] = 0.0
                # print(env.root_states[r_idx, :3])

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            # camera_position += camera_vel * env.dt
            # env.set_camera(camera_position, camera_position + camera_direction)
            camera_position = env.root_states[0, :3].cpu() + camera_vel * 2
            env.set_camera(camera_position, env.root_states[0, :3])

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_vel_abd': env.dof_vel[robot_index, 3*leg_index + 0].item(),
                    'dof_vel_hip': env.dof_vel[robot_index, 3*leg_index + 1].item(),
                    'dof_vel_knee': env.dof_vel[robot_index, 3*leg_index + 2].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'dof_torque_abd': env.torques[robot_index, 3*leg_index + 0].item(),
                    'dof_torque_hip': env.torques[robot_index, 3*leg_index + 1].item(),
                    'dof_torque_knee': env.torques[robot_index, 3*leg_index + 2].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    # 'lf_pos': env._lf_position[robot_index, 2].cpu().numpy(),
                    # 'rf_pos': env._rf_position[robot_index, 2].cpu().numpy(),
                    # 'lf_pos_des': env._lf_position_des[robot_index, 2].cpu().numpy(),
                    # 'rf_pos_des': env._rf_position_des[robot_index, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
def print_obs(obs):
    print("Base Height")
    print("="*50)
    print(obs[0, 0])
    print("="*50)
    print("Base Orientation")
    print("="*50)
    print(obs[0, 1:5])
    print("="*50)
    print("Base Lin vel")
    print("="*50)
    print(obs[0, 5:8])
    print("="*50)
    print("Base Ang vel")
    print("="*50)
    print(obs[0, 8:11])
    print("="*50)
    print("Proj. Grav")
    print("="*50)
    print(obs[0, 11:14])
    print("="*50)
    print("Commands")
    print("="*50)
    print(obs[0, 14:17])
    print("="*50)
    print("dof_pos")
    print("="*50)
    print(obs[0, 17:23])
    print("="*50)
    print("dof_vel")
    print("="*50)
    print(obs[0, 23:29])
    print("="*50)
    print("actions")
    print("="*50)
    print(obs[0, 29:35])
    print("="*50)
    print("phase")
    print("="*50)
    print(obs[0, 35])
    print("="*50)
    print("sin phase")
    print("="*50)
    print(obs[0, 36])
    print("="*50)
    print("cos phase")
    print("="*50)
    print(obs[0, 37])
    print("="*50)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)

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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import yaml

def modify_anymal_envcfg(env_cfg, param_path):
    with open(param_path) as f:
        params = yaml.safe_load(f)
        getattr(env_cfg, 'init_state').default_joint_angles['LF_HAA'] = params['A']
        getattr(env_cfg, 'init_state').default_joint_angles['LH_HAA'] = params['A']
        getattr(env_cfg, 'init_state').default_joint_angles['RF_HAA'] = -params['A']
        getattr(env_cfg, 'init_state').default_joint_angles['RH_HAA'] = -params['A']

        getattr(env_cfg, 'init_state').default_joint_angles['LF_HFE'] = params['H']
        getattr(env_cfg, 'init_state').default_joint_angles['LH_HFE'] = -params['H']
        getattr(env_cfg, 'init_state').default_joint_angles['RF_HFE'] = params['H']
        getattr(env_cfg, 'init_state').default_joint_angles['RH_HFE'] = -params['H']

        getattr(env_cfg, 'init_state').default_joint_angles['LF_KFE'] = -params['K']
        getattr(env_cfg, 'init_state').default_joint_angles['LH_KFE'] = params['K']
        getattr(env_cfg, 'init_state').default_joint_angles['RF_KFE'] = -params['K']
        getattr(env_cfg, 'init_state').default_joint_angles['RH_KFE'] = params['K']
        getattr(env_cfg, 'init_state').pos[2] = params['pos_z']

def modify_mc_envcfg(env_cfg, param_path):
    with open(param_path) as f:
        params = yaml.safe_load(f)
        getattr(env_cfg, 'control').stiffness['joint'] = params['stiffness']
        getattr(env_cfg, 'control').damping['joint'] = params['damping']
        getattr(env_cfg, 'control').action_scale = params['action_scale']
def modify_pat_envcfg(env_cfg, param_path):
    with open(param_path) as f:
        params = yaml.safe_load(f)
        getattr(env_cfg, 'init_state').default_joint_angles['L_hip_joint'] = params['A']
        getattr(env_cfg, 'init_state').default_joint_angles['R_hip_joint'] = -params['A']
        getattr(env_cfg, 'init_state').default_joint_angles['L_thigh_joint'] = params['H']
        getattr(env_cfg, 'init_state').default_joint_angles['R_thigh_joint'] = params['H']
        getattr(env_cfg, 'init_state').default_joint_angles['L_calf_joint'] = params['K']
        getattr(env_cfg, 'init_state').default_joint_angles['R_calf_joint'] = params['K']
        getattr(env_cfg, 'init_state').pos[2] = params['pos_z']
        getattr(env_cfg, 'control').stiffness['joint'] = params['P']
        getattr(env_cfg, 'control').damping['joint'] = params['D']
def train(args):
    # env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    if args.task == 'anymal_c_flat':
        modify_anymal_envcfg(task_registry.get_cfgs(name=args.task)[0], '{}_param.yaml'.format(args.task))
    elif args.task == 'mc_flat':
        modify_mc_envcfg(task_registry.get_cfgs(name=args.task)[0], '{}_param.yaml'.format(args.task))
    elif args.task == 'pat':
        modify_pat_envcfg(task_registry.get_cfgs(name=args.task)[0], '{}_param.yaml'.format(args.task))
    else:
        print('unknown experiment')
        quit()
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # task_registry.set_env_cfg(env_cfg, name=args.task)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    with open('{}_reward.csv'.format(args.task), 'w') as f:
        f.write(str(ppo_runner.final_reward))
if __name__ == '__main__':
    args = get_args()
    train(args)

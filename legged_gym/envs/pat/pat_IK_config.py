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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.pat.pat_config import PatCfg, PatCfgPPO
import math
class PatIKCfg( PatCfg ):
    class experiment:
        recompute_normalization = True
        experiment_name = "pat_pd_no_hist_no_norm"
        n_sample_mean_est = 1000
    class gait():
        swing_time = 0.28
    class foot_placement(PatCfg.foot_placement):
        swing_height = 0.15
        x_default = -0.1
        y_default = 0.012
        z_default = -0.38
    class history:
        history_length = 6
        n_joints = 6
    class init_state( PatCfg.init_state ):
        pos = [0.0, 0.0, 0.45] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'R_hip_joint': 0.3,   # [rad]
            'R_thigh_joint': -0.32,     # [rad]
            'R_calf_joint': 0.83,    # [rad]

            'L_hip_joint': -0.16,   # [rad]
            'L_thigh_joint': -0.29,    # [rad]
            'L_calf_joint': 0.81,     # [rad]
        }

    class env(PatCfg.env):
        num_observations = 40 #76
        num_actions = 6
    class terrain(PatCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
    class control(PatCfg.control ):

        control_type = 'IK'
        stiffness = {'joint': 25.}  # [N*m/rad]
        damping = {'joint': 0.4}     # [N*m*s/rad]
        decimation = 2
        action_scale = 0.1
    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            ori = 0.0
            dof_pos = 0.01
            pos_error = 0.001
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            foot_pos = 0.001

    class commands(PatCfg.commands):
        heading_command = False # if true: compute ang vel command from heading error
        curriculum = False
        resampling_time = 30.
        class ranges(PatCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-2.0, 2.0]
        push_robots = True
        push_interval_s = 0.2
        max_push_vel_xy = 1.

    class rewards( PatCfg.rewards ):
        base_height_target = 0.45
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.8
        tracking_sigma = 1.0
        max_contact_force = 100. # forces above this value are penalized
        class scales:
            tracking_lin_vel = 3.0
            tracking_ang_vel = 3.0
            feet_air_time = 0.3
            slip = -0.08
            foot_clearance = -15.0
            orientation = -3.0
            torques = -6e-4
            base_height = -20.0
            dof_vel = -6e-4
            dof_acc = -0.02
            body_motion = -1.5
            linear_ortho_vel = 0.0
            collision = -1.

    class sim(PatCfg.sim):
        dt =  0.005
class PatIKCfgPPO( PatCfgPPO ):
    class algorithm( PatCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( PatCfgPPO.runner ):
        run_name = ''
        experiment_name = 'pat_pd_no_hist_no_norm'
        load_run = -1 #"Feb11_14-03-54_" # -1 = last run
        checkpoint = -1 #"800" # -1 = last saved model
        max_iterations = 1000

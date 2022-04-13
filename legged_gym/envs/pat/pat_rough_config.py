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
import math
class PatRoughCfg( LeggedRobotCfg ):
    class gait():
        swing_time = 0.33
    class foot_placement():
        swing_height = 0.05
        hight_des = 0.46
        thigh_offset = 0.00
        kappa=-0.077699
        t_prime=0.193597
        alpha = 0.5 #lift swing ratio
        omega = math.sqrt(9.81/hight_des)
        default_foot_loc = [0, 0.06, 0]
        fp_type = 'Donghyun'
        x_step_limit = [-0.2, 0.2]
        y_step_limit = [0.03, 0.2]
    class time_delay():
        sampling_time_range = [0, 0]
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.45] # x,y,z [m]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'L_hip_joint': -0.16,   # [rad]
            'R_hip_joint': 0.3,   # [rad]

            'L_thigh_joint': 0.55,    # [rad]
            'R_thigh_joint': 0.55,     # [rad]

            'L_calf_joint': -0.95,     # [rad]
            'R_calf_joint': -0.95,    # [rad]
        }

    class env( LeggedRobotCfg.env ):
        num_observations = 38
        num_actions = 6
    class terrain( LeggedRobotCfg.terrain ):
        #mesh_type = 'trimesh'
        mesh_type = 'plane'
        measure_heights = False
        rough = False
        curriculum = False
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'TA'
        kpCartesian = 3000
        kdCartesian = 2.0
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 10
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    class commands(LeggedRobotCfg.commands):
        heading_command = False # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5] #[-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5] #[-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pat/urdf/pat.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
        fix_base_link = False
        # collapse_fixed_joints = False
    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5]
        push_robots = True
        push_interval_s = 0.2
        max_push_vel_xy = 1.

    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.45
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        # tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        # soft_dof_vel_limit = 1.
        # soft_torque_limit = 1.
        max_contact_force = 100. # forces above this value are penalized
        class scales( LeggedRobotCfg.rewards.scales ):
            base_height = -20.0
            #penalize pitch and roll
            orientation = -10.0
            #TVR Reward
            # foot_position = -20.0
            #stance foot velocity penality
            foot_velocity = -1.0
            #swing foot GRF penality
            GRF = -0.01

            foot_height_ref = -10
            slip = -0.0003
            joint_motion = -0.000001
            target_smoothness = -0.003
            linear_ortho_vel = 0.75
            body_motion = 1.0

            # feet_air_time =  1.0
            #
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -0.00001
            dof_vel = 0.0
            dof_acc = 0.0
            collision = -5.
            # feet_stumble = -0.0
            # #Penalize policy output change
            action_rate = -0.0
            stand_still = -0.0


class PatRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'pat_rough'
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        max_iterations = 1000

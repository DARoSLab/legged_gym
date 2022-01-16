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

from time import time
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.pat.pat_utils import *

class Pat(LeggedRobot):
    def _custom_init(self, cfg):


        self._body_orientation_des = to_torch(np.zeros((self.num_envs, 4), dtype=np.float32), device=self.device)
        self._body_orientation_des[:, -1] = 1.0
        self._body_omegades = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._body_omegadot_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)

        self._body_position_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._body_vel_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._body_acc_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)

        self._lf_position_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._lf_p_mid_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._lf_vel_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._lf_acc_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)

        self._rf_position_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._rf_p_mid_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._rf_vel_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._rf_acc_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)

        self._lf_position = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._rf_position = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)

        #POLICY NETWORK OUTPUT
        self._Fr = to_torch(np.zeros((self.num_envs, 6, 1), dtype=np.float32), device=self.device)
        self._lf_pf_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)
        self._rf_pf_des = to_torch(np.zeros((self.num_envs, 3), dtype=np.float32), device=self.device)

        #GAIT PARAMETERS
        self._t = to_torch(np.zeros((self.num_envs, 1), dtype=np.float32), device=self.device)
        self._phase = to_torch(np.zeros((self.num_envs, 1), dtype=np.float32), device=self.device)
        self._swing_phases = to_torch(np.zeros((self.num_envs, 2), dtype=np.float32), device=self.device)
        self._swing_states = to_torch(np.zeros((self.num_envs, 2), dtype=np.float32), device=self.device)
        self._prev_swing_states = to_torch(np.zeros((self.num_envs, 2), dtype=np.float32), device=self.device)
        self._stance_to_swining_tans = to_torch(np.zeros((self.num_envs, 2), dtype=np.float32), device=self.device)
        self._tau = to_torch(np.zeros((self.num_envs, 6, 1), dtype=np.float32), device=self.device)
        self._tau_swing = to_torch(np.zeros((self.num_envs, 6, 1), dtype=np.float32), device=self.device)
        self._tau_stance = to_torch(np.zeros((self.num_envs, 6, 1), dtype=np.float32), device=self.device)
        self._swing_time = cfg.gait.swing_time
        self._swing_height = cfg.foot_placement.swing_height
        self._gait_period = 3*self._swing_time #double stance and two single stances
        self._hight_des = to_torch(np.zeros((self.num_envs, 1), dtype=np.float32), device=self.device)
        self._hight_des[:] = cfg.foot_placement.hight_des

        self.start_planning = None
        self.planning_done = None
        # Prepare jacobian tensor
        # For pat, tensor shape is (self.num_envs, 10, 6, 9)
        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "anymal")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)
        if cfg.asset.fix_base_link:
            self._Jc = to_torch(np.zeros((self.num_envs, 6, 6), dtype=np.float32), device=self.device)
        else:
            self._Jc = to_torch(np.zeros((self.num_envs, 6, 12), dtype=np.float32), device=self.device)

        # # Contact Jacobian entries
        # LF_index = 5 #gym.get_asset_rigid_body_dict(pat_asset)["L_foot"]
        # RF_index = 9 #gym.get_asset_rigid_body_dict(pat_asset)["R_foot"]
        #
        # self._j_lf = self.jacobian[:, LF_index - 1, :]
        # self._j_rf = self.jacobian[:, RF_index - 1, :]

        self._rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(self._rb_states)

        self._rb_positions = self.rb_states[:, 0:3].view(self.num_envs, self.num_bodies, 3)
        self._rb_vels = self.rb_states[:, 7:10].view(self.num_envs, self.num_bodies, 3)

        print("Foot Placment Type: {}".format(cfg.foot_placement.fp_type))

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.trunk_idxs = []
        self.lthigh_idxs = []
        self.rthigh_idxs = []
        self.lf_idxs = []
        self.rf_idxs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()

            # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            # start_pose.p = gymapi.Vec3(*self.cfg.init_state.pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

            trunk_idx = self.gym.find_actor_rigid_body_index(env_handle, anymal_handle, "base", gymapi.DOMAIN_SIM)
            self.trunk_idxs.append(trunk_idx)
            lf_idx = self.gym.find_actor_rigid_body_index(env_handle, anymal_handle, "L_foot", gymapi.DOMAIN_SIM)
            self.lf_idxs.append(lf_idx)
            rf_idx = self.gym.find_actor_rigid_body_index(env_handle, anymal_handle, "R_foot", gymapi.DOMAIN_SIM)
            self.rf_idxs.append(rf_idx)
            lthigh_idx = self.gym.find_actor_rigid_body_index(env_handle, anymal_handle, "L_thigh", gymapi.DOMAIN_SIM)
            self.lthigh_idxs.append(lthigh_idx)
            rthigh_idx = self.gym.find_actor_rigid_body_index(env_handle, anymal_handle, "R_thigh", gymapi.DOMAIN_SIM)
            self.rthigh_idxs.append(rthigh_idx)
        # Contact Jacobian entries
        self.LF_index = self.gym.get_asset_rigid_body_dict(robot_asset)["L_foot"]
        self.RF_index = self.gym.get_asset_rigid_body_dict(robot_asset)["R_foot"]



        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        self._rb_properties = self.gym.get_actor_rigid_body_properties(self.envs[0], 0)
        self._rb_masses = to_torch(np.array([getattr(self._rb_properties[i], 'mass') for i in range(self.num_bodies)], dtype=np.float32), device=self.device)
    def compute_observations(self):
        """ Computes observations
        """
        self._body_position = self.rb_states[self.trunk_idxs, :3]
        self._body_orientation = self.rb_states[self.trunk_idxs, 3:7]
        self.obs_buf = torch.cat((  self._body_position[:, 2].view(-1, 1), #body height 1
                                    self._body_orientation, #orientation quat 4
                                    self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 3
                                    self.projected_gravity, # 3
                                    # self.commands[:, :3] * self.commands_scale, #3
                                    self.dof_pos,# 6
                                    self.dof_vel,# 6
                                    self.actions,# 6
                                    self._phase, # 1
                                    ),dim=-1) # 33
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _build_contact_jacobian(self):
        self.gym.refresh_jacobian_tensors(self.sim)
        self._j_lf = self.jacobian[:, self.LF_index - 1, :]
        self._j_rf = self.jacobian[:, self.RF_index - 1, :]
        self._Jc[:, :3, :] = self._j_lf[:, :3, :] #position jacobian only
        self._Jc[:, 3:6, :] = self._j_rf[:, :3, :]
    def _update_gait_info(self):

        self._t = torch.fmod(self._t + self.sim_params.dt, self._gait_period)
        self._phase = self._t/self._gait_period

        # _dstance_idx = (self._phase<(1./3)).squeeze() # double stance
        # _sstance_l_idx = torch.logical_and(torch.logical_not(_dstance_idx), (self._phase<(2./3)).squeeze()) #single stance left foot stance
        # _sstance_r_idx =  (self._phase>(2./3)).squeeze() #right foot stance

        _dstance_idx = (self._phase<0).squeeze() # double stance
        _sstance_l_idx = torch.logical_and(torch.logical_not(_dstance_idx), (self._phase<0.5).squeeze()) #single stance left foot stance
        _sstance_r_idx =  (self._phase>0.5).squeeze() #right foot stance

        self._swing_states[_dstance_idx, 0] = 0.0
        self._swing_states[_dstance_idx, 1] = 0.0

        self._swing_states[_sstance_l_idx, 0] = 1.0
        self._swing_states[_sstance_l_idx, 1] = 0.0

        self._swing_states[_sstance_r_idx, 0] = 0.0
        self._swing_states[_sstance_r_idx, 1] = 1.0

        self._stance_to_swining_tans[self._swing_states > self._prev_swing_states] = 1.0
        self._stance_to_swining_tans[self._swing_states <= self._prev_swing_states] = -1.0

        self._prev_swing_states = torch.clone(self._swing_states)


        # self._swing_phases[_sstance_l_idx, 0] = 3*(self._phase[_sstance_l_idx]-1.0/3).squeeze()
        self._swing_phases[_dstance_idx, 0] = 0.0
        self._swing_phases[_dstance_idx, 1] = 0.0

        self._swing_phases[_sstance_l_idx, 0] = 2*(self._phase[_sstance_l_idx]).squeeze()
        self._swing_phases[_sstance_l_idx, 1] = 0.0

        self._swing_phases[_sstance_r_idx, 0] = 0.0
        # self._swing_phases[_sstance_r_idx, 1] = 3*(self._phase[_sstance_r_idx]-2.0/3).squeeze()
        self._swing_phases[_sstance_r_idx, 1] = 2*(self._phase[_sstance_r_idx]-0.5).squeeze()

    def _donghyun_fp(self):
        t_prime = torch.zeros((2, 1), device=self.device)
        t_prime[:] = self.cfg.foot_placement.t_prime
        kappa = torch.zeros((2, 1), device=self.device)
        kappa[:] = self.cfg.foot_placement.kappa
        omega = self.cfg.foot_placement.omega
        self._body_position = self.rb_states[self.trunk_idxs, :3]

        ll_swing_idx = self._swing_states[:, 0]>0.0
        rl_swing_idx = self._swing_states[:, 1]>0.0
        stance_foot_loc = torch.zeros_like(self._lf_position)
        self._lf_pf_des = torch.zeros_like(self._lf_position)
        self._rf_pf_des = torch.zeros_like(self._lf_position)
        # self._lf_p_mid_des = torch.zeros_like(self._lf_position)
        # self._rf_p_mid_des = torch.zeros_like(self._lf_position)
        stance_foot_loc[ll_swing_idx] = self._rf_position[ll_swing_idx] #right foot stance
        stance_foot_loc[rl_swing_idx] = self._lf_position[rl_swing_idx] #left foot stance


        swing_time_remaining = torch.zeros_like(self._phase)
        swing_time_remaining[rl_swing_idx] = 2*self._swing_time*(1-self._phase[rl_swing_idx])# Right leg stance first
        swing_time_remaining[ll_swing_idx] = self._swing_time*(1-2*self._phase[ll_swing_idx])
        #enable_planning_idx = swing_time_remaining > 0.5 * self.cfg.foot_placement.swing_time
        A = 0.5*((self._com_position[:, :2] - stance_foot_loc[:, :2]) + self._com_vel[:, :2]/omega) #Nx1
        B = 0.5*((self._com_position[:, :2] - stance_foot_loc[:, :2]) - self._com_vel[:, :2]/omega)


        switching_state_pos = (torch.bmm(A.view(-1, 2, 1), torch.exp(omega * swing_time_remaining).view(-1, 1, 1))
                            + torch.bmm(B.view(-1, 2, 1), torch.exp(-omega * swing_time_remaining).view(-1, 1, 1))
                            + stance_foot_loc[:, :2].view(-1, 2, 1))

        switching_state_vel = (omega*torch.bmm(A.view(-1, 2, 1), torch.exp(omega * swing_time_remaining).view(-1, 1, 1))
                                    - torch.bmm(B.view(-1, 2, 1), torch.exp(-omega * swing_time_remaining).view(-1, 1, 1)))

        exp_weight = 1/(omega*torch.tanh(omega * t_prime)) #coth
        target_loc = torch.zeros_like(self._lf_position)

        target_loc[:, :2] = (switching_state_pos*(1-kappa) + switching_state_vel*exp_weight).view(-1, 2)
        target_loc[:, :2] += (self.env_origins[:, :2].view(-1, 2, 1) * kappa).view(-1, 2)
        target_loc[:, 2] = -0.002

        b_positive_sidestep = ll_swing_idx
        target_loc = self._step_length_check(target_loc, b_positive_sidestep, stance_foot_loc)
        self._lf_pf_des[ll_swing_idx] = target_loc[ll_swing_idx]
        self._rf_pf_des[rl_swing_idx] = target_loc[rl_swing_idx]

        # self._lf_p_mid_des[rl_swing_idx, :] = stance_foot_loc[rl_swing_idx, :] #left stance
        # self._lf_p_mid_des[rl_swing_idx, 2] = self.cfg.foot_placement.swing_height #swing height
        # self._rf_p_mid_des[ll_swing_idx, :] = stance_foot_loc[ll_swing_idx] #right stance
        # self._rf_p_mid_des[ll_swing_idx, 2] = self.cfg.foot_placement.swing_height #swing height
        self._lf_p_mid_des[:, 0] = self.env_origins[:, 0] + self.cfg.foot_placement.default_foot_loc[0];
        self._lf_p_mid_des[:, 1] = self.env_origins[:, 1] + self.cfg.foot_placement.default_foot_loc[1];
        self._lf_p_mid_des[:, 2] = self.cfg.foot_placement.swing_height
        self._rf_p_mid_des[:, 0] = self.env_origins[:, 0] + self.cfg.foot_placement.default_foot_loc[0];
        self._rf_p_mid_des[:, 1] = self.env_origins[:, 1] + -self.cfg.foot_placement.default_foot_loc[1];
        self._rf_p_mid_des[:, 2] = self.cfg.foot_placement.swing_height

    def _step_length_check(self, target_loc, b_positive_sidestep, stance_foot):
        # X limit check

        x_step_length_limit_ = torch.zeros((2,1), device=self.device)
        x_step_length_limit_[0] = -0.1
        x_step_length_limit_[1] = 0.1

        y_step_length_limit_ = torch.zeros((2,1), device=self.device)
        y_step_length_limit_[0] = 0.03 #min
        y_step_length_limit_[1] = 0.12 #max

        #X LIMIT CHECK
        x_step_length = target_loc[:, 0] - stance_foot[:, 0]
        x_min_idx = x_step_length < x_step_length_limit_[0]
        x_max_idx = x_step_length > x_step_length_limit_[1]

        target_loc[x_min_idx, 0] = stance_foot[x_min_idx, 0] + x_step_length_limit_[0]
        target_loc[x_max_idx, 0] = stance_foot[x_max_idx, 0] + x_step_length_limit_[1]

        #Y limit check
        y_step_length =  target_loc[:, 1] - stance_foot[:, 1]

        y_min_idx = y_step_length < y_step_length_limit_[0]
        y_max_idx = y_step_length > y_step_length_limit_[1]

        mv_left_min_idx = torch.logical_and(b_positive_sidestep, y_min_idx)
        mv_left_max_idx = torch.logical_and(b_positive_sidestep, y_max_idx)

        target_loc[mv_left_min_idx, 1] = stance_foot[mv_left_min_idx, 1] + y_step_length_limit_[0]
        target_loc[mv_left_min_idx, 1] = stance_foot[mv_left_min_idx, 1] + y_step_length_limit_[1]

        mv_right_min_idx = torch.logical_and(torch.logical_not(b_positive_sidestep), y_min_idx)
        mv_right_max_idx = torch.logical_and(torch.logical_not(b_positive_sidestep), y_max_idx)

        target_loc[mv_right_min_idx, 1] = stance_foot[mv_right_min_idx, 1] - y_step_length_limit_[0]
        target_loc[mv_right_min_idx, 1] = stance_foot[mv_right_min_idx, 1] - y_step_length_limit_[1]

        return target_loc

    def _capture_point_fp(self):
        _vBody = self.rb_states[self.trunk_idxs, 7:10]
        _lthigh_position = self.rb_states[self.lthigh_idxs, :3]
        _rthigh_position = self.rb_states[self.rthigh_idxs, :3]

        ll_swing_idx = self._swing_states[:, 0]>0.0
        rl_swing_idx = self._swing_states[:, 1]>0.0

        #update pf only at the beginning of swing
        trans_idx = self._stance_to_swining_tans>0.0
        ltrans_idx = trans_idx[:, 0]
        rtrans_idx = trans_idx[:, 1]

        self._body_vel_des[:, :2] = self.commands[:, :2] * self.commands_scale[:2]
        self._body_vel_des[:, 2] = 0.0
        #  Raibert heuristi
        self._lf_pf_des[ltrans_idx]  = _lthigh_position[ltrans_idx] + _vBody[ltrans_idx]*self._swing_time*0.5
        #  capture-point
        self._lf_pf_des[ltrans_idx]  += torch.sqrt(self._hight_des/9.8)[ltrans_idx]*(self._body_vel_des[ltrans_idx]-_vBody[ltrans_idx])
        # zero foot height
        self._lf_pf_des[:, 2] = 0
        self._lf_pf_des[:, 1] += self.cfg.foot_placement.thigh_offset
        #  Raibert heuristi
        self._rf_pf_des[rtrans_idx]  = _rthigh_position[rtrans_idx] + _vBody[rtrans_idx]*self._swing_time*0.5
        #  capture-point
        self._rf_pf_des[rtrans_idx]  += torch.sqrt(self._hight_des/9.8)[rtrans_idx]*(self._body_vel_des[rtrans_idx]-_vBody[rtrans_idx])
        self._rf_pf_des[:, 2] = 0
        self._rf_pf_des[:, 1] -= self.cfg.foot_placement.thigh_offset
    def _update_foot_placement(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg.foot_placement.fp_type=='CP':
            self._capture_point_fp()
        else:
            self._donghyun_fp()
    def _update_com_state(self):
        self._com_position = torch.sum(self._rb_positions*self._rb_masses.view(1, self.num_bodies, 1), dim=1)/torch.sum(self._rb_masses)
        self._com_vel = torch.sum(self._rb_vels*self._rb_masses.view(1, self.num_bodies, 1), dim=1)/torch.sum(self._rb_masses)
    def _compute_swing_trajectory(self, fp_algorithm='cp'):

        self._lf_position = self.rb_states[self.lf_idxs, :3]
        self._lf_vel = self.rb_states[self.lf_idxs, 7:10]
        self._rf_position = self.rb_states[self.rf_idxs, :3]
        self._rf_vel = self.rb_states[self.rf_idxs, 7:10]
        ll_swing_idx = self._swing_states[:, 0]>0.0 #left leg swing
        rl_swing_idx = self._swing_states[:, 1]>0.0 #right leg swing
        if(self.cfg.foot_placement.fp_type=='CP'):
            self._lf_position_des[ll_swing_idx], self._lf_vel_des[ll_swing_idx], self._lf_acc_des[ll_swing_idx] = computeHeightSwingTrajectory(self._lf_position[ll_swing_idx],
            self._lf_pf_des[ll_swing_idx],
            self._swing_phases[ll_swing_idx, 0],
            self._swing_height,
            self._swing_time)

            self._rf_position_des[rl_swing_idx], self._rf_vel_des[rl_swing_idx], self._rf_acc_des[rl_swing_idx] = computeHeightSwingTrajectory(self._rf_position[rl_swing_idx],
            self._rf_pf_des[rl_swing_idx],
            self._swing_phases[rl_swing_idx, 1],
            self._swing_height,
            self._swing_time)
        else:
            self._lf_position_des[ll_swing_idx], self._lf_vel_des[ll_swing_idx], self._lf_acc_des[ll_swing_idx] = computeLiftSwingTrajectory(self._lf_position[ll_swing_idx],
            self._lf_p_mid_des[ll_swing_idx],
            self._lf_pf_des[ll_swing_idx],
            self._swing_phases[ll_swing_idx, 0],
            alpha = self.cfg.foot_placement.alpha,
            swing_time=self.cfg.gait.swing_time)

            self._rf_position_des[rl_swing_idx], self._rf_vel_des[rl_swing_idx], self._rf_acc_des[rl_swing_idx] = computeLiftSwingTrajectory(self._rf_position[rl_swing_idx],
            self._rf_p_mid_des[rl_swing_idx],
            self._rf_pf_des[rl_swing_idx],
            self._swing_phases[rl_swing_idx, 1],
            alpha = self.cfg.foot_placement.alpha,
            swing_time=self.cfg.gait.swing_time)


    def _swing_impedence_control(self):

        lfootForce = self.cfg.control.kpCartesian * (self._lf_position_des - self._lf_position)
        lfootForce += self.cfg.control.kdCartesian * (self._lf_vel_des - self._lf_vel)
        rfootForce = self.cfg.control.kpCartesian * (self._rf_position_des - self._rf_position)
        rfootForce += self.cfg.control.kdCartesian * (self._rf_vel_des - self._rf_vel_des)

        ll_stance_idx = self._swing_states[:, 0]<=0.0 #left leg swing
        rl_stance_idx = self._swing_states[:, 1]<=0.0 #left leg swing


        lfootForce[ll_stance_idx, :] = 0 #N z component
        rfootForce[rl_stance_idx, :] = 0

        swingfootForce = torch.cat([lfootForce, rfootForce], dim=1)

        self._tau_swing = torch.bmm(self._Jc.transpose(1, 2), swingfootForce.unsqueeze(2))

    def _stance_jt_control(self, actions_scaled):

        ll_stance_idx = self._swing_states[:, 0]<=0.0 #left leg stance
        rl_stance_idx = self._swing_states[:, 1]<=0.0 #right leg stance

        #principle of vertical impulse scaling
        # x y
        self._Fr[ll_stance_idx, :2, 0] = actions_scaled[ll_stance_idx, :2] #N x,y component
        self._Fr[rl_stance_idx, 3:5, 0] = actions_scaled[rl_stance_idx, 3:5] #N x,y component

        self._Fr[ll_stance_idx, 2, 0] = -8.606*9.8*2.0 + actions_scaled[ll_stance_idx, 2] #N z component
        self._Fr[rl_stance_idx, 5, 0] = -8.606*9.8*2.0 + actions_scaled[rl_stance_idx, 5]
        # swing reaction force 0
        self._Fr[torch.logical_not(ll_stance_idx), :3] = 0 #N z component
        self._Fr[torch.logical_not(rl_stance_idx), 3:] = 0
        self._tau_stance = torch.bmm(self._Jc.transpose(1, 2), self._Fr)
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        elif control_type == "J": #jacobian transpose
            self._build_contact_jacobian()
            self._update_gait_info()
            self._update_com_state()
            self._update_foot_placement()
            self._compute_swing_trajectory()
            self._swing_impedence_control()
            self._stance_jt_control(actions_scaled)
            if self.cfg.asset.fix_base_link:
                torques = (self._tau_swing + self._tau_stance).view(self.torques.shape)
            else:
                torques = (self._tau_swing + self._tau_stance)[:, 6:, 0].contiguous()
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """

        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact

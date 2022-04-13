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
import csv

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.envs import Pat
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.pat.pat_utils import *
from legged_gym.utils.pat_terrain import PatTerrain
class PatIK(Pat):

    def compute_observations(self):
        """ Computes observations
        """
        self._body_position = self.rb_states[self.trunk_idxs, :3]
        self._body_orientation = self.root_states[:, 3:7]
        self._update_history()
        self._update_foot_position()

        n_sample_mean_est = self.cfg.experiment.n_sample_mean_est
        # obs_buf_un = torch.cat((    self._body_orientation, #orientation quat 4
        #                             self.base_ang_vel*self.obs_scales.ang_vel, # 3
        #                             self.commands[:, :3]* self.commands_scale, #3
        #                             (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,# 6
        #                             self.dof_vel*self.obs_scales.dof_vel,# 6
        #                             # self._jointPosErrorHist[:, (self._historyLength-6)*self._nJoints : (self._historyLength-5)*self._nJoints],#6
        #                             # self._jointPosErrorHist[:, (self._historyLength-4)*self._nJoints : (self._historyLength-3)*self._nJoints],#6
        #                             # self._jointPosErrorHist[:, (self._historyLength-2)*self._nJoints : (self._historyLength-1)*self._nJoints],#6
        #                             # self._jointVelHist[:, (self._historyLength-6)*self._nJoints : (self._historyLength-5)*self._nJoints],#6
        #                             # self._jointVelHist[:, (self._historyLength-4)*self._nJoints : (self._historyLength-3)*self._nJoints],#6
        #                             # self._jointVelHist[:, (self._historyLength-2)*self._nJoints : (self._historyLength-1)*self._nJoints],#6
        #                             (self._prepreviousAction - self.default_dof_pos)*self.obs_scales.dof_pos,#6
        #                             (self._previousAction - self.default_dof_pos)*self.obs_scales.dof_pos,#6
        #                             self._foot_pos, #6
        #                             #self._base_phase,#1
        #                             #torch.sin(self._phases),#2
        #                             #torch.cos(self._phases)#2
        #                             ),dim=-1) # 76
        obs_buf_un = torch.cat((    self._body_orientation, #orientation quat 4
                                    self.base_ang_vel, # 3
                                    self.commands[:, :3], #3
                                    self.dof_pos,# 6
                                    self.dof_vel,# 6
                                    # self._jointPosErrorHist[:, (self._historyLength-6)*self._nJoints : (self._historyLength-5)*self._nJoints],#6
                                    # self._jointPosErrorHist[:, (self._historyLength-4)*self._nJoints : (self._historyLength-3)*self._nJoints],#6
                                    # self._jointPosErrorHist[:, (self._historyLength-2)*self._nJoints : (self._historyLength-1)*self._nJoints],#6
                                    # self._jointVelHist[:, (self._historyLength-6)*self._nJoints : (self._historyLength-5)*self._nJoints],#6
                                    # self._jointVelHist[:, (self._historyLength-4)*self._nJoints : (self._historyLength-3)*self._nJoints],#6
                                    # self._jointVelHist[:, (self._historyLength-2)*self._nJoints : (self._historyLength-1)*self._nJoints],#6
                                    (self._prepreviousAction - self.default_dof_pos)*self.obs_scales.dof_pos,#6
                                    (self._previousAction - self.default_dof_pos)*self.obs_scales.dof_pos,#6
                                    self._foot_pos, #6
                                    #self._base_phase,#1
                                    #torch.sin(self._phases),#2
                                    #torch.cos(self._phases)#2
                                    ),dim=-1) # 76

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            obs_buf_un = torch.cat((obs_buf_un, heights), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        if self.enable_mean_var_est:
            if(self._iter == 0):
                self.obs_buf_mean_cummulative = obs_buf_un.mean(dim=0)
                self.obs_buf_var_cummulative  = obs_buf_un.var(dim=0)
            elif n_sample_mean_est>0:
                    if(self._iter < n_sample_mean_est): #estrimate mean and var using 10 batchs
                        alpha = (self._iter - 1)/(self._iter)
                        self.obs_buf_mean_cummulative  = alpha*self.obs_buf_mean_cummulative  + (1-alpha)*obs_buf_un.mean(dim=0)
                        self.obs_buf_var_cummulative  = alpha*self.obs_buf_var_cummulative  + (1-alpha)*obs_buf_un.var(dim=0)
                    elif(self._iter == n_sample_mean_est):
                        # np.savetxt("/home/dan/DARo/config/mean.csv", self.obs_buf_mean_cummulative.cpu().numpy().reshape(-1), delimiter=",")
                        # np.savetxt("/home/dan/DARo/config/var.csv", self.obs_buf_mean_cummulative.cpu().numpy().reshape(-1), delimiter=",")
                        torch.save(self.obs_buf_mean_cummulative.cpu(), self.mean_path)
                        torch.save(self.obs_buf_var_cummulative.cpu(), self.var_path)
                    else:
                        #use the same mean and var for the rest
                        pass
            else:
                alpha = (self._iter - 1)/(self._iter)
                self.obs_buf_mean_cummulative  = alpha*self.obs_buf_mean_cummulative  + (1-alpha)*obs_buf_un.mean(dim=0)
                self.obs_buf_var_cummulative  = alpha*self.obs_buf_var_cummulative  + (1-alpha)*obs_buf_un.var(dim=0)
                torch.save(self.obs_buf_mean_cummulative.cpu(), self.mean_path)
                torch.save(self.obs_buf_var_cummulative.cpu(), self.var_path)
                # print("mean_var_cummulative: ", self.obs_buf_mean_cummulative)
        # print("_iter: ", self._iter)
        # print("dof_pos: ", self.dof_pos_mean_cummulative)
        # self.obs_buf_mean_cummulative[:] = 0.0
        # self.obs_buf_mean_cummulative[:] = 1.0
        self.obs_buf = (obs_buf_un-self.obs_buf_mean_cummulative)/torch.sqrt(self.obs_buf_var_cummulative+1e-8)
        self.obs_buf = self.obs_buf.clip(-10.0, 10.0)
        # print()
        # print("Qun:", obs_buf_un[0, :])
        # print("Q:", self.obs_buf[0, :])
        if(self._iter==0 and self.enable_mean_var_est): #discard first iter data its useless
            self.obs_buf_mean_cummulative[:] = 0.0
            self.obs_buf_var_cummulative[:]  = 0.0
        # print("Obs mean: ", self.obs_buf.mean(dim=0))
        # print("Obs var: ", self.obs_buf.var(dim=0))
        self._iter += 1
        # add perceptive inputs if not blind
        # add noise if needed
    def _update_foot_position(self): #base to foot in world frame
        self._foot_pos = (self.rb_states.view(self.num_envs, -1, 13)[:, self.feet_indices, :3] - self.root_states[:, :3].view(-1, 1, 3)).view(-1, 6)
        # print("foot_pos_b: ", self._foot_pos[0])
        # print("foot_pos_w: ", self.rb_states.view(self.num_envs, -1, 13)[0, self.feet_indices, :3])
        # print("root_pos_w: ", self.root_states[0, :3])
    def _update_history(self):
        self._historyTempMem[:] = self._jointVelHist[:]
        self._jointVelHist[:, :(self._historyLength-1)*self._nJoints] = self._historyTempMem[:, self._nJoints: ]
        self._jointVelHist[:, (self._historyLength-1)*self._nJoints:] = self.dof_vel
        self._historyTempMem[:] = self._jointPosErrorHist[:]
        self._jointPosErrorHist[:, :(self._historyLength-1)*self._nJoints] = self._historyTempMem[:, self._nJoints: ]
        self._jointPosErrorHist[:, (self._historyLength-1)*self._nJoints:] = self._joint_target - self.dof_pos
        self._prepreviousAction = self._previousAction
        self._previousAction = self._joint_target

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """


        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        _nJoints = 6
        noise_vec[:4] = noise_scales.ori * noise_level
        noise_vec[4:7] =  noise_scales.ang_vel * noise_level
        noise_vec[7:10] = 0. # commands
        noise_vec[10:10 + _nJoints] = noise_scales.dof_pos * noise_level
        noise_vec[10 + _nJoints:10 + 2*_nJoints] = noise_scales.dof_vel * noise_level
        noise_vec[10 + 2*_nJoints:10 + 5*_nJoints] = noise_scales.pos_error * noise_level
        noise_vec[10 + 5*_nJoints:10 + 8*_nJoints] = noise_scales.dof_vel * noise_level
        noise_vec[70:76] = noise_scales.foot_pos * noise_level # previous actions

        if self.cfg.terrain.measure_heights:
            noise_vec[76:254] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    def _update_gait_info(self):
        #Left Leg swing First
        # TODO: Replace with delta_phases with policy output

        # self._base_phase = 2*math.pi*torch.fmod(self._t + self.sim_params.dt, self._gait_period)/self._gait_period
        self._t = torch.fmod(self._t + self.sim_params.dt, self._gait_period)
        self._base_phase = 2*math.pi*self._t/self._gait_period
        self._delta_phases[:, 0] = 0
        self._delta_phases[:, 1] = math.pi
        self._phases = torch.fmod(self._base_phase + self._delta_phases, 2*math.pi)
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
            self._joint_target = actions_scaled + self.default_dof_pos
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        elif control_type == "IK":
            torques = self._compute_IK_torques(actions_scaled)
        elif control_type == "TA":
            torques = self._compute_actual_torques(actions_scaled)
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
    def _compute_IK_torques(self, actions_scaled):
        _motorTauMax = torch.tensor([1.5, 2, 2, 1.5, 2, 2], device=self.device)
        _batteryV = 24;
        _motorKT = torch.tensor([0.068, 0.091, 0.091, 0.068, 0.091, 0.091], device=self.device)
        _motorR = torch.tensor([0.262, 0.173, 0.173, 0.262, 0.173, 0.173], device=self.device)
        _jointDamping = .01;
        _jointDryFriction = .2;
        _gr = torch.zeros((1, 6)).to(device="cuda")
        for leg in range(2):
            _gr[0, 3*leg] = 6 #abduction
            _gr[0, 3*leg+1] = 9 #hip
            _gr[0, 3*leg+2] = 14.49 #Knee

        self._update_gait_info()
        q_ref = swing_ref3d(self._phases,
                            self.device,
                            x_default=self.cfg.foot_placement.x_default,
                            y_default=self.cfg.foot_placement.y_default,
                            z_default = self.cfg.foot_placement.z_default,
                            swing_height=self.cfg.foot_placement.swing_height)

        # self._joint_target = actions_scaled + q_ref
        self._joint_target = q_ref
        tauDes = self.p_gains*(self._joint_target - self.dof_pos) - self.d_gains*self.dof_vel
        # print(tauDes.shape)
        tauDesMotor = torch.div(tauDes, _gr);        # motor torque
        iDes = tauDesMotor / (_motorKT * 1.5);  # i = tau / KT
        # bemf =  qd * _gr * _motorKT * 1.732;     # back emf
        bemf = self.dof_vel * _gr * _motorKT * 2.;       # back emf
        vDes = iDes * _motorR + bemf;          # v = I*R + emf
        vActual = torch.clip(vDes, -_batteryV, _batteryV);  # limit to battery voltage
        tauActMotor =1.5 * _motorKT * (vActual - bemf) / _motorR;  # tau = Kt * I = Kt * V / R
        tauAct = torch.clip(tauActMotor, -self.torque_limits, self.torque_limits)*_gr;
        tauAct = tauAct - _jointDamping * self.dof_vel - _jointDryFriction * torch.sign(self.dof_vel);
        return tauAct
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)


        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()


    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-1.5*ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_GRF(self):
        """
        penalize swing foot GRF
        """
        grf_n = self.contact_forces[:, self.feet_indices, :].norm(dim=1)
        return (((grf_n > 0.1) & (self._phases < math.pi))*grf_n).sum(dim=1)

    def _reward_foot_velocity(self):
        """
        Penalize stance foot velocity
        """
        vel_n = self.rb_states.view(self.num_envs, -1, 3)[:, self.feet_indices, 7:10].norm(dim=1)
        return (((vel_n > 0.001) & (self._phases >= math.pi))*vel_n).sum(dim=1)

    def _reward_linear_ortho_vel(self):
        """
        From anymal paper
        Penalizes the velocity orthogonal to the target direction
        """
        tmp = torch.bmm(self.commands[:, :2].view(-1, 1, 2), self.base_lin_vel[:, :2].view(-1, 2, 1)).view(-1, 1)
        vo = self.base_lin_vel[:, :2] - tmp*self.commands[:, :2]
        return torch.exp(-3.0*torch.square(torch.norm(vo, dim=1)))
    def _reward_body_motion(self):
        """
        Penalizes the body velocity in directions not part of the command
        """
        return 0.8*torch.square(self.base_lin_vel[:, 2]) + 0.4*torch.abs(self.base_ang_vel[:, 0]) + 0.4*torch.abs(self.base_ang_vel[:, 1])
    def _reward_joint_motion(self):
        """
        Penalizes joint velocity and acceleration to avoid vibrations
        """
        dof_acc = (self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        return -0.75*torch.sum(0.01*torch.square(self.dof_vel) + torch.square(dof_acc), dim=1)
    def _reward_target_smoothness(self):
        """
        The magnitude of the first and second order finite dif-
        ference derivatives of the target foot positions are penalized such that the generated foot
        trajectories become smoother
        """
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1) +\
              torch.sum(torch.square(self.actions - 2*self.last_actions + self.last_last_actions), dim=1)
    def _reward_slip(self):
        """
        Penalize the foot velocity if the foot is in contact with the ground to reduce slippage
        Independent of gait
        """
        foot_contact = self.rb_states.view(self.num_envs, -1, 13)[:, self.feet_indices, 2] < 0.04
        foot_velocity_n = self.rb_states.view(self.num_envs, -1, 13)[:, self.feet_indices, 7:9].norm(dim=2)
        return (foot_contact*torch.pow(foot_velocity_n, 2)).sum(dim=1)
    def _reward_foot_clearance(self):
        """
        encourages foot clearance 0.05 m scaled with velocity
        """
        foot_velocity_n = self.rb_states.view(self.num_envs, -1, 13)[:, self.feet_indices, 7:9].norm(dim=2)
        foot_z = self.rb_states.view(self.num_envs, -1, 13)[:, self.feet_indices, 2]
        return (torch.square(foot_z - 0.05)*torch.sqrt(foot_velocity_n)).sum(dim=1)
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)), dim=1)
    def _reward_ik_ref_tracking(self):
        return (self.dof_pos - self._joint_target).square().sum(dim=1)

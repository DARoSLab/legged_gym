"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Pat Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Pat robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *


import math
import numpy as np
import torch
import random
import time

import matplotlib.pyplot as plt
#pytorch3d https://github.com/facebookresearch/pytorch3d
from pytorch3d.transforms import quaternion_to_matrix, so3_log_map
from legged_gym.envs.pat.pat_config import PatCfg

print(PatCfg.init_state.default_joint_angles)
def cubicBezier(y0, yf, x):
    dim = y0.shape[1]
    yDiff = (yf - y0).view(-1, dim, 1)
    bezier = (x * x * x + 3. * (x * x * (1. - x))).view(-1, 1, 1)

    return (y0 + torch.bmm(yDiff, bezier).squeeze(-1)).view(-1, dim)

# Cubic bezier interpolation derivative between y0 and yf.  x is between 0 and 1
def cubicBezierFirstDerivative(y0, yf, x):
    dim = y0.shape[1]
    yDiff = (yf - y0).view(-1, dim, 1)
    bezier = (6. * x * (1. - x)).view(-1, 1, 1)
    return (y0 + torch.bmm(yDiff, bezier).squeeze(-1)).view(-1, dim)
# Cubic bezier interpolation second derivative between y0 and yf.  x is between 0 and 1
def cubicBezierSecondDerivative(y0, yf, x):
    dim = y0.shape[1]
    yDiff = (yf - y0).view(-1, dim, 1)
    bezier = (6. - 12.* x).view(-1, 1, 1)
    return (y0 + torch.bmm(yDiff, bezier).squeeze(-1)).view(-1, dim)

def computeSwingTrajectory(_p0, _pf, phase):
    global _swing_height, _swing_time

    _p = cubicBezier(_p0, _pf, phase)
    _pdot = cubicBezierFirstDerivative(_p0, _pf, phase) / _swing_time
    _pddot = cubicBezierSecondDerivative(_p0, _pf, phase) / (_swing_time*_swing_time)

    _lift_idx = phase<0.5
    _p[_lift_idx, 2] = cubicBezier(_p0[_lift_idx, 2].view(-1, 1), _p0[_lift_idx, 2].view(-1, 1) + _swing_height, phase[_lift_idx] * 2).squeeze(-1)
    _pdot[_lift_idx, 2]= cubicBezierFirstDerivative(_p0[_lift_idx, 2].view(-1, 1), _p0[_lift_idx, 2].view(-1, 1) + _swing_height, phase[_lift_idx] * 2).squeeze(-1) * 2 / _swing_time
    _pddot[_lift_idx, 2] = cubicBezierSecondDerivative(_p0[_lift_idx, 2].view(-1, 1), _p0[_lift_idx, 2].view(-1, 1) + _swing_height, phase[_lift_idx] * 2).squeeze(-1) * 4 / (_swing_time * _swing_time)

    _step_idx = torch.logical_not(_lift_idx)

    _p[_step_idx, 2] = cubicBezier(_p0[_step_idx, 2].view(-1, 1) + _swing_height, _pf[_step_idx, 2].view(-1, 1), phase[_step_idx] * 2 - 1).squeeze(-1)
    _pdot[_step_idx, 2] = cubicBezierFirstDerivative(_p0[_step_idx, 2].view(-1, 1) + _swing_height, _pf[_step_idx, 2].view(-1, 1), phase[_step_idx] * 2 - 1).squeeze(-1) * 2 / _swing_time
    _pddot[_step_idx, 2] = cubicBezierSecondDerivative(_p0[_step_idx, 2].view(-1, 1) + _swing_height, _pf[_step_idx, 2].view(-1, 1), phase[_step_idx] * 2 - 1).squeeze(-1) * 4 / (_swing_time * _swing_time)

    return _p, _pdot, _pddot

def computeLiftSwingTrajectory(_p0, _p_mid, _pf, swing_phase, alpha=0.5):
    global _swing_time

    _lift_idx = swing_phase<alpha
    _step_idx = torch.logical_not(_lift_idx)

    _p = torch.zeros_like(_p0)
    _pdot = torch.zeros_like(_p0)
    _pddot = torch.zeros_like(_p0)

    _p[_lift_idx, :] = cubicBezier(_p0[_lift_idx, :], _p_mid[_lift_idx, :], swing_phase[_lift_idx]/alpha)
    _pdot[_lift_idx, :] = cubicBezierFirstDerivative(_p0[_lift_idx, :], _p_mid[_lift_idx, :], swing_phase[_lift_idx]) / (alpha*_swing_time)
    _pddot[_lift_idx, :] = cubicBezierSecondDerivative(_p0[_lift_idx, :], _p_mid[_lift_idx, :], swing_phase[_lift_idx]) / (alpha*alpha*_swing_time*_swing_time)

    _p[_step_idx, :] = cubicBezier(_p_mid[_step_idx, :], _pf[_step_idx, :], (swing_phase[_step_idx]-alpha)/(1.0-alpha))
    _pdot[_step_idx, :] = cubicBezierFirstDerivative(_p_mid[_step_idx, :], _pf[_step_idx, :], (swing_phase[_step_idx]-alpha)/(1.0-alpha)) / (alpha*_swing_time)
    _pddot[_step_idx, :] = cubicBezierSecondDerivative(_p_mid[_step_idx, :], _pf[_step_idx, :], (swing_phase[_step_idx]-alpha)/(1.0-alpha)) / ((1-alpha)*(1-alpha)*_swing_time*_swing_time)

    return _p, _pdot, _pddot

def _update_gait_info():
    global _phase, _swing_states, sim_params, _gait_period, _t, _stance_to_swining_tans, _prev_swing_states
    _t = torch.fmod(_t + sim_params.dt, _gait_period)
    _phase = _t/_gait_period

    # _dstance_idx = (_phase<(1./3)).squeeze() # double stance
    # _sstance_l_idx = torch.logical_and(torch.logical_not(_dstance_idx), (_phase<(2./3)).squeeze()) #single stance left foot stance
    # _sstance_r_idx =  (_phase>(2./3)).squeeze() #right foot stance

    _dstance_idx = (_phase<0).squeeze() # double stance
    _sstance_l_idx = torch.logical_and(torch.logical_not(_dstance_idx), (_phase<0.5).squeeze()) #single stance left foot stance
    _sstance_r_idx =  (_phase>0.5).squeeze() #right foot stance

    _swing_states[_dstance_idx, 0] = 0.0
    _swing_states[_dstance_idx, 1] = 0.0

    _swing_states[_sstance_l_idx, 0] = 1.0
    _swing_states[_sstance_l_idx, 1] = 0.0

    _swing_states[_sstance_r_idx, 0] = 0.0
    _swing_states[_sstance_r_idx, 1] = 1.0

    _stance_to_swining_tans[_swing_states > _prev_swing_states] = 1.0
    _stance_to_swining_tans[_swing_states <= _prev_swing_states] = -1.0

    _prev_swing_states = torch.clone(_swing_states)

    _swing_phases[_dstance_idx, 0] = 0.0
    _swing_phases[_dstance_idx, 1] = 0.0


    # _swing_phases[_sstance_l_idx, 0] = 3*(_phase[_sstance_l_idx]-1.0/3).squeeze()
    # _swing_phases[_sstance_l_idx, 1] = 0.0
    #
    # _swing_phases[_sstance_r_idx, 0] = 0.0
    # _swing_phases[_sstance_r_idx, 1] = 3*(_phase[_sstance_r_idx]-2.0/3).squeeze()

    _swing_phases[_sstance_l_idx, 0] = 2*(_phase[_sstance_l_idx]).squeeze()
    _swing_phases[_sstance_l_idx, 1] = 0.0

    _swing_phases[_sstance_r_idx, 0] = 0.0
    _swing_phases[_sstance_r_idx, 1] = 2*(_phase[_sstance_r_idx]-0.5).squeeze()




def _compute_swing_trajectory():
    global _lf_position, _rf_position, _lf_p_mid_des, _lf_pf_des,\
            _rf_p_mid_des, _rf_pf_des, _t_swing, _phase, _swing_states, _swing_times, \
            _lf_position_des, _lf_vel_des, _lf_acc_des, _swing_period, _phase

    # ll_swing_idx = _swing_states[:, 0]>0.0 #left leg swing
    # _lf_position_des[ll_swing_idx], _lf_vel_des[ll_swing_idx], _lf_acc_des[ll_swing_idx] = computeSwingTrajectory(_lf_position[ll_swing_idx],
    #                             _lf_pf_des[ll_swing_idx],
    #                             _swing_phases[ll_swing_idx, 0])
    #
    # rl_swing_idx = _swing_states[:, 1]>0.0 #right leg swing
    # _rf_position_des[rl_swing_idx], _rf_vel_des[rl_swing_idx], _rf_acc_des[rl_swing_idx] = computeSwingTrajectory(_rf_position[rl_swing_idx],
    #                             _rf_pf_des[rl_swing_idx],
    #                             _swing_phases[rl_swing_idx, 1])

    ll_swing_idx = _swing_states[:, 0]>0.0 #left leg swing
    _lf_position_des[ll_swing_idx], _lf_vel_des[ll_swing_idx], _lf_acc_des[ll_swing_idx] = computeLiftSwingTrajectory(_lf_position[ll_swing_idx],_lf_p_mid_des[ll_swing_idx],
                                _lf_pf_des[ll_swing_idx],
                                _swing_phases[ll_swing_idx, 0])

    rl_swing_idx = _swing_states[:, 1]>0.0 #right leg swing
    _rf_position_des[rl_swing_idx], _rf_vel_des[rl_swing_idx], _rf_acc_des[rl_swing_idx] = computeLiftSwingTrajectory(_rf_position[rl_swing_idx],_rf_p_mid_des[rl_swing_idx],
                                _rf_pf_des[rl_swing_idx],
                                _swing_phases[rl_swing_idx, 1])

def _swing_impedence_control():
    global _lf_position, _rf_position, _lf_p_mid_des, _lf_pf_des,\
            _rf_p_mid_des, _rf_pf_des, _t_swing, _phase, _swing_states, _swing_times, \
            _lf_position_des, _lf_vel_des, _lf_acc_des, _swing_period, _phase, _tau_swing,\
            _Jc, lf_hist_des

    kpCartesian = 2000.0
    kdCartesian = 2.0

    lfootForce = kpCartesian * (_lf_position_des - _lf_position)
    lfootForce += kdCartesian * (_lf_vel_des - _lf_vel)
    rfootForce = kpCartesian * (_rf_position_des - _rf_position)
    rfootForce += kdCartesian * (_rf_vel_des - _rf_vel_des)

    ll_stance_idx = _swing_states[:, 0]<=0.0 #left leg swing
    rl_stance_idx = _swing_states[:, 1]<=0.0 #left leg swing
    z = float(_lf_position_des[0, 2])
    lf_hist_des.append(z)
    # print('left z_des: ', _lf_position_des[0, 2], _lf_position[0, 2], _swing_states[0, 0])
    # print('right z_des: ', _rf_position_des[0, 2], _rf_position[0, 2], _swing_states[0, 1])

    lfootForce[ll_stance_idx, :] = 0 #N z component
    rfootForce[rl_stance_idx, :] = 0

    swingfootForce = torch.cat([lfootForce, rfootForce], dim=1)


    # tau_lf = torch.bmm(_j_lf[:, :3, :3].transpose(1, 2), lfootForce.unsqueeze(2))

    # tau_rf = torch.bmm(_j_rf[:, :3, 3:].transpose(1, 2), rfootForce.unsqueeze(2))
    _tau_swing = torch.bmm(_Jc.transpose(1, 2), swingfootForce.unsqueeze(2))

    # _tau_swing = torch.cat([tau_lf, tau_rf], dim=1)


def _stance_jt_control():
    global _tau_stance, _swing_states, _Fr, _Jc

    ll_stance_idx = _swing_states[:, 0]<=0.0 #left leg stance
    rl_stance_idx = _swing_states[:, 1]<=0.0 #right leg stance
    # swing reaction force 0
    _Fr[torch.logical_not(ll_stance_idx), :] = 0 #N z component
    _Fr[torch.logical_not(rl_stance_idx), :] = 0

    #principle of vertical impulse scaling
    #
    _Fr[ll_stance_idx, 2] = -8.606*9.8*2.0 #N z component
    _Fr[rl_stance_idx, 5] = -8.606*9.8*2.0

    _tau_stance = torch.bmm(_Jc.transpose(1, 2), _Fr)

def _capture_point_fp():
    pass
def _compute_com():
    global _rb_positions, _rb_vels, _rb_masses, _com_position, _com_vel, _num_bodies
    _com_position = torch.sum(_rb_positions[:, :_num_bodies, :]*_rb_masses[:_num_bodies].view(1, _num_bodies, 1), dim=1)/torch.sum(_rb_masses)
    _com_vel = torch.sum(_rb_vels[:, :_num_bodies, :]*_rb_masses[:_num_bodies].view(1, _num_bodies, 1), dim=1)/torch.sum(_rb_masses)


def _donghyun_fp(continous_update=True):
    global device, _lf_pf_des, _rf_pf_des, _swing_states, _lf_position, _rf_position,\
           _lthigh_position, _rthigh_position, _vBody, _hight_des, _body_vel_des, _kp, _kd, \
           _body_position, _com_position, _com_vel, _swing_time, device, _phase, _lf_p_mid_des,\
            _rf_p_mid_des, _swing_height, _kappa, _t_prime

    omega_ = PatCfg.foot_placement.omega
    ll_swing_idx = _swing_states[:, 0]>0.0
    rl_swing_idx = _swing_states[:, 1]>0.0

    stance_foot_loc = torch.zeros_like(_lf_position)
    _lf_pf_des = torch.zeros_like(_lf_position)
    _rf_pf_des = torch.zeros_like(_lf_position)

    stance_foot_loc[ll_swing_idx] = _rf_position[ll_swing_idx] #right foot stance
    stance_foot_loc[rl_swing_idx] = _lf_position[rl_swing_idx] #left foot stance
    # _lf_p_mid = torch.zeros_like(_lf_position)
    # _rf_p_mid = torch.zeros_like(_lf_position)
    _lf_p_mid_des[:, 0] = _body_position[:, 0] + PatCfg.foot_placement.default_foot_loc[0];
    _lf_p_mid_des[:, 1] = _body_position[:, 1] + PatCfg.foot_placement.default_foot_loc[1];
    _lf_p_mid_des[:, 2] = PatCfg.foot_placement.swing_height;
    _rf_p_mid_des[:, 0] = _body_position[:, 0] + PatCfg.foot_placement.default_foot_loc[0];
    _rf_p_mid_des[:, 1] = _body_position[:, 1] +-PatCfg.foot_placement.default_foot_loc[1];
    _rf_p_mid_des[:, 2] = PatCfg.foot_placement.swing_height;

    # _lf_p_mid_des[rl_swing_idx, :] = stance_foot_loc[rl_swing_idx, :] #left stance
    # _lf_p_mid_des[rl_swing_idx, 2] = _swing_height #swing height
    # _rf_p_mid_des[ll_swing_idx, :] = stance_foot_loc[ll_swing_idx] #right stance
    # _rf_p_mid_des[ll_swing_idx, 2] = _swing_height #swing height

    N = _lf_position.shape[0]
    swing_time_remaining = torch.zeros((N, 1), device=device)
    swing_time_remaining[rl_swing_idx] = 2*_swing_time*(1-_phase[rl_swing_idx])# Right leg stance first
    swing_time_remaining[ll_swing_idx] = _swing_time*(1-2*_phase[ll_swing_idx])
    # print("_phase: ", _phase[0])
    # print("com: ", _com_position[0])
    # print("com_vel: ", _com_vel[0])
    # print("swing_time: ", swing_time_remaining[0])


    A = 0.5*((_com_position[:, :2] - stance_foot_loc[:, :2]) + _com_vel[:, :2]/omega_) #Nx1
    B = 0.5*((_com_position[:, :2] - stance_foot_loc[:, :2]) - _com_vel[:, :2]/omega_)


    switching_state_pos = (torch.bmm(A.view(-1, 2, 1), torch.exp(omega_ * swing_time_remaining).view(-1, 1, 1))
                        + torch.bmm(B.view(-1, 2, 1), torch.exp(-omega_ * swing_time_remaining).view(-1, 1, 1))
                        + stance_foot_loc[:, :2].view(-1, 2, 1))

    switching_state_vel = (omega_*torch.bmm(A.view(-1, 2, 1), torch.exp(omega_ * swing_time_remaining).view(-1, 1, 1))
                                - torch.bmm(B.view(-1, 2, 1), torch.exp(-omega_ * swing_time_remaining).view(-1, 1, 1)))


    exp_weight = 1/(omega_*torch.tanh(omega_ * _t_prime)) #coth


    target_loc = torch.zeros_like(_lf_position)
    target_loc[:, :2] = (switching_state_pos*(1-_kappa) + switching_state_vel * exp_weight).view(-1, 2)
    target_loc[:, 2] = -0.002

    b_positive_sidestep = ll_swing_idx

    target_loc = _step_length_check(target_loc, b_positive_sidestep, stance_foot_loc)
    _lf_pf_des[ll_swing_idx] = target_loc[ll_swing_idx]
    _rf_pf_des[rl_swing_idx] = target_loc[rl_swing_idx]
    # print("_lf_pf_des: ", _lf_pf_des[0], _lf_position[0])
    # print("_rf_pf_des: ", _rf_pf_des[0], _rf_position[0])
def _step_length_check(target_loc, b_positive_sidestep, stance_foot):
    global device
    # X limit check

    x_step_length_limit_ = torch.zeros((2,1), device=device)
    x_step_length_limit_[0] = PatCfg.foot_placement.x_step_limit[0]
    x_step_length_limit_[1] = PatCfg.foot_placement.x_step_limit[1]

    y_step_length_limit_ = torch.zeros((2,1), device=device)
    y_step_length_limit_[0] = PatCfg.foot_placement.y_step_limit[0]
    y_step_length_limit_[1] = PatCfg.foot_placement.y_step_limit[1]

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


def _update_foot_placement():
    global device, _lf_pf_des, _rf_pf_des, _swing_states, _lf_position, _rf_position,\
           _lthigh_position, _rthigh_position, _vBody, _hight_des, _body_vel_des
    ll_swing_idx = _swing_states[:, 0]>0.0
    rl_swing_idx = _swing_states[:, 1]>0.0

    #update pf only at the beginning of swing
    # _lf_pf_des = torch.clone(_lf_position)
    # if(_stance_to_swining_tans[0, 0] > 0.0):
    #     print("left leg transition detected")
    # else:
    #     print("#")

    # % Footsetp heuristics in world frame
    # H_r_foot = [RotYaw*r_hip(1:3,foot),...                                            % hip location from COM in World Frame
    # T_s(foot,k)/2*RotYaw_input*[x_input(7:8,k);0],...                                           % forward stepping heuristic
    # T_s(foot,k)/2*CrossProd([0;0;x_input(12,k)])*(RotYaw*r_hip(1:3,foot)),...  % turning in place heuristic
    # sqrt(x(3)/norm_g)*(x(7:9,k) - RotYaw_input*x_input(7:9,k)),...                          % capture point heuristic
    # x(3,k)/norm_g*CrossProd(RotYaw_input*[x_input(7:8,k);0])*[0;0;x_input(12,k)]];            % High Speed Turning
    # H_X(i_r+1:i_r+3) = P'*P*(H_r_foot*K_Hr(1:size(H_r_foot,2),k));
    #stance to swing transition
    trans_idx = _stance_to_swining_tans>0.0
    ltrans_idx = trans_idx[:, 0]
    rtrans_idx = trans_idx[:, 1]


    #  Raibert heuristi
    _lf_pf_des[ltrans_idx]   = _lthigh_position[ltrans_idx] + _vBody[ltrans_idx]*_swing_time*0.5
    #  capture-point
    _lf_pf_des[ltrans_idx]  += torch.sqrt(_hight_des/9.8)[ltrans_idx]*(_vBody[ltrans_idx]-_body_vel_des[ltrans_idx])
    # zero foot height
    _lf_pf_des[:, 2] = 0

    _lf_pf_des[ltrans_idx, 1] += 0.06

    #  Raibert heuristi
    _rf_pf_des[rtrans_idx]  = _rthigh_position[rtrans_idx] + _vBody[rtrans_idx]*_swing_time*0.5
    #  capture-point
    _rf_pf_des[rtrans_idx]  += torch.sqrt(_hight_des/9.8)[rtrans_idx]*(_vBody[rtrans_idx]-_body_vel_des[rtrans_idx])
    _rf_pf_des[:, 2] = 0
    _rf_pf_des[rtrans_idx, 1] -= 0.06

def _make_floating_base_MM():
    #     0.136759  -0.00280711    0.0108167  -7.4186e-11     0.816169  -0.00910761
    #  -0.00280711     0.133311  0.000483809    -0.816169  1.86265e-09    -0.163369
    #    0.0108167  0.000483807     0.045424    0.0091076     0.163369  1.86265e-09
    # -1.81756e-09    -0.816169   0.00910762        8.606  9.11377e-10 -1.91714e-10
    #     0.816169 -6.62037e-10     0.163369 -9.11377e-10        8.606            0
    #   -0.0091076    -0.163369  1.98769e-09  1.91714e-10            0        8.606
    global _A_float_base, device
    _A_float_base[:, :3, :3] =  8.606*torch.eye(3, device=device)
    _A_float_base[:, 3,  3] =  0.136759
    _A_float_base[:, 4,  4] =  0.133311
    _A_float_base[:, 5,  5] =  0.045424

def _update_full_MM():
    global _A_float_base, _A_act, _A
    _A[:, :6, :6] = _A_float_base
    _A[:, 6:, 6:] = _A_act

def _WeightedInverse(J, Winv, threshold = 0.0001):
  tmp = torch.matmul(J, torch.matmul(Winv, J.transpose(1, 2)))
  lambda_inv = torch.linalg.pinv(tmp, rcond=threshold)
  return torch.matmul(Winv, torch.matmul(J.transpose(1, 2), lambda_inv))

def _update_task_jacobians():
    global device, num_envs, _body_orientation, _omegaBody, _vBody, \
           _pos_task_jacobian, _ori_task_jacobian, \
           _lf_task_jacobian, _rf_task_jacobian
    _R = quaternion_to_matrix(_body_orientation)
    omega_cross_xdot = torch.cross(_omegaBody, _vBody)
    _R_T = torch.transpose(_R, 1, 2)
    _ori_task_jacobian[:, :3, :3] = _R_T
    _pos_task_jacobian[:, :3, 3:6] = _R
    _pos_task_jdotqdot = torch.bmm(_R_T, omega_cross_xdot.unsqueeze(2)).squeeze()
    _lf_task_jacobian = _j_lf[:, :3, :]
    _rf_task_jacobian = _j_rf[:, :3, :]
    #_ori_task_jdotqdot = 0


def _update_ops_command():
    global num_envs, device,\
           _body_orientation_des, _body_omega_des, _body_omegadot_des,\
           _body_position_des, _body_vel_des, _body_acc_des,\
           _body_orientation, _body_position,\
           _pos_error, _pos_op_cmd, _ori_err_so3, _omega_des, _omega_err, _ori_op_cmd,\
           _lf_position_des, _lf_vel_des, _lf_acc_des, _lf_position, _lf_vel,\
           _lf_pos_error, _lf_op_cmd, \
           _rf_position_des, _rf_vel_des, _rf_acc_des, _rf_position, _rf_vel, \
           _rf_pos_error, _rf_op_cmd


    #Body Position Task
    _vBody_tmp = torch.bmm(_R_T, _vBody.unsqueeze(2)).squeeze()
    _pos_error  = _body_position_des - _body_position
    _pos_op_cmd = _Kp_pos*(_body_position_des - _body_position) + \
                  _Kd_pos*(_body_vel_des-_vBody_tmp) + \
                  _body_acc_des
    #Body Orientation Task
    ori_inv = quat_conjugate(_body_orientation)
    ori_err = quat_mul(_body_orientation_des, ori_inv)
    ori_err[ori_err[:, -1] < 0]*=-1
    _ori_err_so3 = so3_log_map(quaternion_to_matrix(ori_err))

    _omega_des = torch.bmm(_R_T, _body_omega_des.unsqueeze(2)).squeeze()
    _omega_err = torch.bmm(_R_T, (_body_omega_des - _omegaBody).unsqueeze(2)).squeeze()

    _ori_op_cmd = _Kp_ori*_ori_err_so3 + \
                  _Kd_ori*_omega_err + \
                  _body_omegadot_des

    #left foot position task
    _lf_pos_error = _Kp_kin*(_lf_position_des-_lf_position)
    _lf_op_cmd = _Kp_swing*_lf_pos_error + \
                 _Kd_swing*(_lf_vel_des-_lf_vel) +\
                 _lf_acc_des
    #right foot position task
    _rf_pos_error = _Kp_kin*(_rf_position_des-_rf_position)
    _rf_op_cmd = _Kp_swing*_rf_pos_error + \
                 _Kd_swing*(_rf_vel_des-_rf_vel)+ \
                 _rf_acc_des
def _build_contact_jacobian():
    global _Jc, _JcDotQdot, _j_lf, _j_rf
    # if _j_lf.shape[2]==12:
    _Jc[:, :3, :] = _j_lf[:, :3, :] #position jacobian only
    _Jc[:, 3:6, :] = _j_rf[:, :3, :]
    # _JcDotQdot = np.zeros(()) set to zero for now
def _compute_wbc():
    global num_envs, device, _Jc, _JcDotQdot, _A, _eye, _qddot_cmd, _qdot_cmd_act, _q_cmd_act
    _build_contact_jacobian()
    Ainv_ = torch.inverse(_A)
    # #Contact Constraint
    JcBar = _WeightedInverse(_Jc, Ainv_)
    _qddot_cmd = torch.bmm(JcBar, -_JcDotQdot.unsqueeze(2))
    Npre = _eye - torch.bmm(JcBar, _Jc)
    delta_q = to_torch(np.zeros((num_envs, 12, 1), dtype=np.float32), device=device)
    _qdot_cmd = to_torch(np.zeros((num_envs, 12, 1), dtype=np.float32), device=device)
    #Body position
    Jt = _pos_task_jacobian
    JtDotQdot = _pos_task_jdotqdot
    xddot = _pos_op_cmd
    pos_error = _pos_error.unsqueeze(2)
    vel_des = _body_vel_des.unsqueeze(2)


    JtPre = torch.bmm(Jt, Npre)
    JtBar = _WeightedInverse(JtPre, Ainv_)


    # Kinematics
    delta_q = torch.bmm(JtBar, pos_error  - torch.bmm(Jt, delta_q))
    _qdot_cmd = torch.bmm(JtBar, vel_des  - torch.bmm(Jt,_qdot_cmd))

    # #Acceleration
    _qddot_cmd += torch.bmm(JtBar, xddot.unsqueeze(2) - JtDotQdot.unsqueeze(2) - torch.bmm(Jt, _qddot_cmd))
    Npre = torch.bmm(Npre,  -torch.bmm(JtBar, JtPre)+_eye)

    #Body Orientation
    Jt = _ori_task_jacobian
    JtDotQdot = _ori_task_jdotqdot
    xddot = _pos_op_cmd
    pos_error = _ori_err_so3.unsqueeze(2)
    vel_des = _omega_des.unsqueeze(2)

    JtPre = torch.bmm(Jt, Npre)
    JtBar = _WeightedInverse(JtPre, Ainv_)


    # Kinematics
    delta_q = torch.bmm(JtBar, pos_error  - torch.bmm(Jt, delta_q))
    _qdot_cmd = torch.bmm(JtBar, vel_des  - torch.bmm(Jt,_qdot_cmd))

    # #Acceleration
    _qddot_cmd += torch.bmm(JtBar, xddot.unsqueeze(2) - JtDotQdot.unsqueeze(2) - torch.bmm(Jt, _qddot_cmd))
    Npre = torch.bmm(Npre,  -torch.bmm(JtBar, JtPre)+_eye)

    #Left foot
    Jt = _lf_task_jacobian
    JtDotQdot = _j_lfdotqdot
    xddot = _lf_op_cmd
    pos_error = _lf_pos_error.unsqueeze(2)
    vel_des = _lf_vel_des.unsqueeze(2)

    JtPre = torch.bmm(Jt, Npre)
    JtBar = _WeightedInverse(JtPre, Ainv_)


    # Kinematics
    delta_q = torch.bmm(JtBar, pos_error  - torch.bmm(Jt, delta_q))
    _qdot_cmd = torch.bmm(JtBar, vel_des  - torch.bmm(Jt,_qdot_cmd))


    # #Acceleration
    _qddot_cmd += torch.bmm(JtBar, xddot.unsqueeze(2) - JtDotQdot.unsqueeze(2) - torch.bmm(Jt, _qddot_cmd))
    Npre = torch.bmm(Npre,  -torch.bmm(JtBar, JtPre)+_eye)
    #Right foot
    Jt = _rf_task_jacobian
    JtDotQdot = _j_rfdotqdot
    xddot = _rf_op_cmd
    pos_error = _rf_pos_error.unsqueeze(2)
    vel_des = _rf_vel_des.unsqueeze(2)

    JtPre = torch.bmm(Jt, Npre)
    JtBar = _WeightedInverse(JtPre, Ainv_)


    # Kinematics
    delta_q = torch.bmm(JtBar, pos_error  - torch.bmm(Jt, delta_q))
    _qdot_cmd = torch.bmm(JtBar, vel_des  - torch.bmm(Jt,_qdot_cmd))

    # #Acceleration
    _qddot_cmd += torch.bmm(JtBar, xddot.unsqueeze(2) - JtDotQdot.unsqueeze(2) - torch.bmm(Jt, _qddot_cmd))
    Npre = torch.bmm(Npre,  -torch.bmm(JtBar, JtPre)+_eye)
    _qdot_cmd_act = _qddot_cmd[:, 6:, :]
    _q_cmd_act = (delta_q[:, 6:, :] + _dof_pos)

def _compute_torque_command():
    global _A, _Jc, _Fr, _qddot_cmd, _Kp_joint, _Kd_joint, _tau

    tau_ff = (torch.bmm(_A, _qddot_cmd) - torch.bmm(_Jc.transpose(1,2), _Fr))[:, 6:, :]
    _tau = tau_ff + _Kp_joint*(_q_cmd_act-_dof_pos) + _Kd_joint*(_qdot_cmd_act-_dof_vel)
def _debug_viz():
    global _lf_pf_des, _lf_position
    gym.clear_lines(viewer)
    ll_swing_idx = _swing_states[:, 0]>0.0 #left leg swing
    sphase = np.linspace(0, 1, 100)
    pts = []
    for sp in sphase:
        _swing_phases[ll_swing_idx, 0] = sp
        p, _, _ = computeSwingTrajectory(_lf_position[ll_swing_idx],
                                             _lf_pf_des[ll_swing_idx],
                                             _swing_phases[ll_swing_idx, 0])
        pts.append(p.cpu().numpy())
    if(len(pts[0])>0):
        pts = np.array(pts)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(num_envs):
            if i==0:
                ax.plot3D(pts[:, i, 0], pts[:, i, 1], pts[:, i, 2], 'red')
            else:
                ax.plot3D(pts[:, i, 0], pts[:, i, 1], pts[:, i, 2], 'gray')
        plt.show()
        quit()

    # if(pts[0].shape[0]>0):
    #     for e_idx in range(len(envs)):
    #         verts = []
    #         colors = []
    #         print(_body_position[ll_swing_idx])
    #         print(_lf_pf_des[ll_swing_idx])
    #
    #         line = torch.concat([_body_position[ll_swing_idx][e_idx].view(-1, 1),
    #                             _lf_pf_des[ll_swing_idx][e_idx].view(-1, 1)], dim=1).cpu().numpy()
    #         print('verts: ', line)
    #         verts.append(line)
    #         # for i in range(1, 100):
    #         #     line = torch.concat([pts[i-1][e_idx].view(-1, 1),pts[i][e_idx].view(-1, 1)], dim=1).cpu().numpy()
    #         #     verts.append(line)
    #             # colors.append(np.random.uniform(0, 1, (3)))
    #         # colors.append(np.random.uniform(0, 1, (3)))
    #         colors.append([1, 0, 0])
    #         gym.add_lines(viewer, envs[e_idx], 1, np.array(verts, dtype=np.float32), np.array(colors, dtype=np.float32))
    #
    # pass

np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# Add custom arguments
args = gymutil.parse_arguments(description="Pat Tensor OSC Example",
                               custom_parameters=[
                                   {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
                                   {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                   {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])

num_envs = args.num_envs
# set torch device
# device = args.sim_device if args.use_gpu_pipeline else 'cpu'
device = 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 0.005 #1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = False #args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = False #args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "../../resources/robots/"

# load pat asset
pat_asset_file = "pat/urdf/pat.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = PatCfg.asset.fix_base_link
asset_options.disable_gravity = False
asset_options.flip_visual_attachments = True
asset_options.collapse_fixed_joints = True
pat_asset = gym.load_asset(sim, asset_root, pat_asset_file, asset_options)


# configure pat dofs
pat_dof_props = gym.get_asset_dof_properties(pat_asset)
pat_lower_limits = pat_dof_props["lower"]
pat_upper_limits = pat_dof_props["upper"]
pat_ranges = pat_upper_limits - pat_lower_limits

_num_bodies = gym.get_asset_rigid_body_count(pat_asset)
print("num bodies: {}".format(_num_bodies))
body_names = gym.get_asset_rigid_body_names(pat_asset)
print("body names:", body_names)

# use torque drive for all dofs
pat_dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
pat_dof_props["stiffness"].fill(0.0)
pat_dof_props["damping"].fill(0.0)


# default dof states and position targets
pat_num_dofs = gym.get_asset_dof_count(pat_asset)
default_dof_pos = np.zeros(pat_num_dofs, dtype=np.float32)
default_dof_pos[0] = PatCfg.init_state.default_joint_angles['L_hip_joint']
default_dof_pos[1] = PatCfg.init_state.default_joint_angles['L_thigh_joint']
default_dof_pos[2] = PatCfg.init_state.default_joint_angles['L_calf_joint']
default_dof_pos[3] = PatCfg.init_state.default_joint_angles['R_hip_joint']
default_dof_pos[4] = PatCfg.init_state.default_joint_angles['R_thigh_joint']
default_dof_pos[5] = PatCfg.init_state.default_joint_angles['R_calf_joint']
default_dof_state = np.zeros(pat_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos
# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# configure env grid
# num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

pat_pose = gymapi.Transform()
pat_pose.p = gymapi.Vec3(0, 0, PatCfg.init_state.pos[2])


lf_des_pose = gymapi.Transform()
lf_des_pose.p = gymapi.Vec3(0, 0.3, 0.0)
rf_des_pose = gymapi.Transform()
rf_des_pose.p = gymapi.Vec3(0, 0.3, 0.0)

# quit()
# print(dir(pat_pose))
# quit()
envs = []
trunk_idxs = []
lthigh_idxs = []
rthigh_idxs = []
lf_idxs = []
rf_idxs = []
init_pos_list = []
init_rot_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)






# _init_root_state.pose = pat_pose
# _init_root_state.vel = pat_vel


# _init_root_state[:3]
lf_sphere_handles = []
rf_sphere_handles = []
pat_handles = []
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add pat
    pat_handle = gym.create_actor(env, pat_asset, pat_pose, "pat", i, 2)
    pat_handles.append(pat_handle)
    # set dof properties
    gym.set_actor_dof_properties(env, pat_handle, pat_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, pat_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, pat_handle, default_dof_pos)

    trunk_idx = gym.find_actor_rigid_body_index(env, pat_handle, "base", gymapi.DOMAIN_SIM)
    trunk_idxs.append(trunk_idx)
    lf_idx = gym.find_actor_rigid_body_index(env, pat_handle, "L_foot", gymapi.DOMAIN_SIM)
    lf_idxs.append(lf_idx)
    rf_idx = gym.find_actor_rigid_body_index(env, pat_handle, "R_foot", gymapi.DOMAIN_SIM)
    rf_idxs.append(rf_idx)
    lthigh_idx = gym.find_actor_rigid_body_index(env, pat_handle, "L_thigh", gymapi.DOMAIN_SIM)
    lthigh_idxs.append(lthigh_idx)
    rthigh_idx = gym.find_actor_rigid_body_index(env, pat_handle, "R_thigh", gymapi.DOMAIN_SIM)
    rthigh_idxs.append(rthigh_idx)

    lf_des_sphere = gym.create_sphere(sim, 0.02, None)
    lf_des_sphere_handle = gym.create_actor(env, lf_des_sphere, lf_des_pose, "lf_des", i, 0)
    gym.set_rigid_body_color(env, lf_des_sphere_handle, 0, gymapi.MESH_NONE, gymapi.Vec3(0.0, 0.5, 0.0))
    lf_sphere_handles.append(lf_des_sphere_handle)
    #no collision with foot
    props = gym.get_actor_rigid_shape_properties(envs[i], lf_sphere_handles[i])
    props[0].filter = 2 # 2 && 2 == 1 !=0 mask collision with foot
    gym.set_actor_rigid_shape_properties(envs[i], lf_sphere_handles[i], props)

    rf_des_sphere = gym.create_sphere(sim, 0.02, None)
    rf_des_sphere_handle = gym.create_actor(env, rf_des_sphere, rf_des_pose, "rf_des", i, 0)
    gym.set_rigid_body_color(env, rf_des_sphere_handle, 0, gymapi.MESH_NONE, gymapi.Vec3(0.0, 0.0, 0.5))
    rf_sphere_handles.append(rf_des_sphere_handle)
    #no collision with foot
    props = gym.get_actor_rigid_shape_properties(envs[i], rf_des_sphere_handle)
    props[0].filter = 2 # 2 && 2 == 1 !=0 mask collision with foot
    gym.set_actor_rigid_shape_properties(envs[i], rf_des_sphere_handle, props)

# _init_root_state_tensor = to_torch(np.zeros((num_envs, 13)), device=device)
# _init_root_state_tensor[:, 2] = 0.44
# _init_root_state_tensor[:, 6] = 1.0
# _init_root_state_tensor[:, 8] = -0.09
# gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(_init_root_state_tensor))

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)
gym.prepare_sim(sim)

# Prepare jacobian tensor
# For pat, tensor shape is (num_envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "pat")
jacobian = gymtorch.wrap_tensor(_jacobian)

print("jacobian: ", _jacobian.shape)
# Contact Jacobian entries
LF_index = gym.get_asset_rigid_body_dict(pat_asset)["L_foot"]
RF_index = gym.get_asset_rigid_body_dict(pat_asset)["R_foot"]

_j_lf = jacobian[:, LF_index - 1, :]
_j_rf = jacobian[:, RF_index - 1, :]

print("_j_lf: ", _j_lf.shape, type(_j_lf))

# Prepare mass matrix tensor
# For pat, tensor shape is (num_envs, 9, 9)
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "pat")
_A_act = gymtorch.wrap_tensor(_massmatrix)
#
_ori_task_jacobian = to_torch(np.zeros((num_envs, 3, pat_num_dofs+6),
                                        dtype=np.float32), device=device)
_pos_task_jacobian = to_torch(np.zeros((num_envs, 3, pat_num_dofs+6),
                                        dtype=np.float32), device=device)
_lf_task_jacobian = to_torch(np.zeros((num_envs, 3, pat_num_dofs+6),
                                        dtype=np.float32), device=device)
_rf_task_jacobian = to_torch(np.zeros((num_envs, 3, pat_num_dofs+6),
                                        dtype=np.float32), device=device)

_ori_task_jdotqdot = to_torch(np.zeros((num_envs, 3),
                                        dtype=np.float32), device=device)
_pos_task_jdotqdot = to_torch(np.zeros((num_envs, 3),
                                        dtype=np.float32), device=device)



# Rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

_rb_properties = gym.get_actor_rigid_body_properties(envs[0], 0)

_rb_masses = to_torch(np.array([getattr(_rb_properties[i], 'mass') for i in range(_num_bodies)], dtype=np.float32), device=device)

_rb_positions = rb_states[:, 0:3].view(num_envs, -1, 3)
_rb_vels = rb_states[:, 7:10].view(num_envs, -1, 3)
# DOF state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
_dof_vel = dof_states[:, 1].view(num_envs, 6, 1)
_dof_pos = dof_states[:, 0].view(num_envs, 6, 1)

_Kp_pos = 100.0
_Kd_pos = 10.0

_Kp_ori = 100.0
_Kd_ori = 10.0

_Kp_swing = 100.0
_Kd_swing = 10.0

_Kp_kin  = 3.0

_Kp_joint = 100.0
_Kd_joint = 1.0

_kp = to_torch(np.zeros((3, 1), dtype=np.float32), device=device)
_kd = to_torch(np.zeros((3, 1), dtype=np.float32), device=device)
_kp[0] = 1.1
_kp[1] = 1.8
_kd[0] = 0.3
_kd[1] = 0.3


_body_orientation_des = to_torch(np.zeros((num_envs, 4), dtype=np.float32), device=device)
_body_orientation_des[:, -1] = 1.0
_body_omega_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_body_omegadot_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)

_body_position_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_body_vel_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_body_acc_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)

_lf_position_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_lf_vel_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_lf_acc_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)

_rf_position_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_rf_vel_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_rf_acc_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)


_R = to_torch(np.zeros((num_envs, 3, 3), dtype=np.float32), device=device)
_R_T = to_torch(np.zeros((num_envs, 3, 3), dtype=np.float32), device=device)
if asset_options.fix_base_link:
    _Jc = to_torch(np.zeros((num_envs, 6, 6), dtype=np.float32), device=device)
else:
    _Jc = to_torch(np.zeros((num_envs, 6, 12), dtype=np.float32), device=device)
_JcDotQdot = to_torch(np.zeros((num_envs, 6), dtype=np.float32), device=device)

_j_lfdotqdot = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_j_rfdotqdot = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)



#MASS MATRIX
_A_float_base = to_torch(np.zeros((num_envs, 6, 6), dtype=np.float32), device=device)
_A = to_torch(np.zeros((num_envs, 12, 12), dtype=np.float32), device=device)
_eye = to_torch(np.eye(12, dtype=np.float32), device=device)
_make_floating_base_MM()

#POLICY NETWORK OUTPUT
_Fr = to_torch(np.zeros((num_envs, 6, 1), dtype=np.float32), device=device)
_pf_des = to_torch(np.zeros((num_envs, 3, 1), dtype=np.float32), device=device)

_lf_p_mid_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_lf_pf_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_rf_p_mid_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)
_rf_pf_des = to_torch(np.zeros((num_envs, 3), dtype=np.float32), device=device)

#GAIT PARAMETERS
_t = to_torch(np.zeros((num_envs, 1), dtype=np.float32), device=device)
_phase = to_torch(np.zeros((num_envs, 1), dtype=np.float32), device=device)
_swing_phases = to_torch(np.zeros((num_envs, 2), dtype=np.float32), device=device)
_swing_states = to_torch(np.zeros((num_envs, 2), dtype=np.float32), device=device)
_prev_swing_states = to_torch(np.zeros((num_envs, 2), dtype=np.float32), device=device)
_stance_to_swining_tans = to_torch(np.zeros((num_envs, 2), dtype=np.float32), device=device)
_tau = to_torch(np.zeros((num_envs, 6, 1), dtype=np.float32), device=device)
_tau_swing = to_torch(np.zeros((num_envs, 6, 1), dtype=np.float32), device=device)
_tau_stance = to_torch(np.zeros((num_envs, 6, 1), dtype=np.float32), device=device)
# _swing_time = 0.5
# _swing_height = 0.15
# _gait_period = 3*_swing_time #double stance and two single stances
_swing_time = PatCfg.gait.swing_time
_swing_height = PatCfg.foot_placement.swing_height
_gait_period = 2*_swing_time #two single stances
_t_prime = torch.zeros((2, 1), device=device)
_t_prime[:] = PatCfg.foot_placement.t_prime
_kappa = torch.zeros((2, 1), device=device)
_kappa[:] = PatCfg.foot_placement.kappa

_hight_des = to_torch(np.zeros((num_envs, 1), dtype=np.float32), device=device)
_hight_des[:] = PatCfg.init_state.pos[2]
_body_vel_des[:, 0] = 0.0
_body_vel_des[:, 1] = 0.0
# SIM LOOP
sim_count = 0
lf_hist = []
lf_hist_des = []
_lf_des_tensor = to_torch(np.zeros((num_envs, 13)), device=device)
rf_scale = -0.01
logger = {}
logger['_lf_position']=[]
logger['_rf_position']=[]
logger['_lf_position_des']=[]
logger['_rf_position_des']=[]
logger['_swing_states']=[]
logger['_Fr']=[]
logger['_com_position']=[]
logger['_com_vel']=[]
logger['_phase'] = []
logger['_t_stamp'] = []
step_count = 0
log_data = False
while step_count < 100 and not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    _body_position = rb_states[trunk_idxs, :3]
    _body_orientation = rb_states[trunk_idxs, 3:7]

    _vBody = rb_states[trunk_idxs, 7:10]
    _omegaBody = rb_states[trunk_idxs, 10:13]
    _lf_position = rb_states[lf_idxs, :3]
    _lf_vel = rb_states[lf_idxs, 7:10]
    _rf_position = rb_states[rf_idxs, :3]
    _rf_vel = rb_states[rf_idxs, 7:10]



    _lthigh_position = rb_states[lthigh_idxs, :3]
    _rthigh_position = rb_states[rthigh_idxs, :3]

    _update_gait_info()
    _compute_com()
    _donghyun_fp()
    # _update_foot_placement()
    _compute_swing_trajectory()
    _build_contact_jacobian()
    _swing_impedence_control()
    _stance_jt_control()
    # lf_hist.append(_lf_position[0, 2].numpy())
    #
    # sim_count += 1
    # if sim_count > 1000:
    #
    #     plt.plot(np.array(lf_hist))
    #     plt.plot(np.array(lf_hist_des))
    #     plt.show()
    #     break
    # _update_task_jacobians()
    # _update_ops_command()
    # _update_full_MM()
    # _compute_wbc()
    # _compute_torque_command()
    # _debug_viz()
    # print(_tau_swing[0, 2], _tau_stance[0, 2])
    if asset_options.fix_base_link:
        _tau  = _tau_swing + _tau_stance
    else:
        _tau  = (_tau_swing + _tau_stance)[:, 6:, :].contiguous()
    # _tau  = (_tau_swing + _tau_stance)
    # if sim_count < 1000:
    #     # set forces and force positions for ant root bodies (first body in each env)
    #     forces = torch.zeros((num_envs, _num_bodies, 3), device=device, dtype=torch.float)
    #     force_positions = _rb_positions.clone()
    #     if sim_count%5 == 0:
    #         forces[:, 0, 2] = 200
    #     force_positions[:, 0, 2] += 0.0
    #     gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)
        # print("iter [{}] applying [{}] Nm of force".format(sim_count, forces[0, 0, 2]))
    verts = np.zeros((4, 3), dtype=np.float32)
    colors = np.zeros((6, 1), dtype=np.float32)
    colors[1] = 0.5
    colors[5] = 0.5
    for i in range(num_envs):
        state = gym.get_actor_rigid_body_states(envs[i], lf_sphere_handles[i], gymapi.STATE_NONE)
        state['pose']['p'].fill((_lf_position_des[i, 0], _lf_position_des[i, 1], _lf_position_des[i, 2]))
        state['pose']['r'].fill((0, 0, 0, 1))
        state['vel']['linear'].fill((0, 0, 0))
        state['vel']['angular'].fill((0, 0, 0))
        gym.set_actor_rigid_body_states(envs[i], lf_sphere_handles[i], state, gymapi.STATE_ALL)
        state = gym.get_actor_rigid_body_states(envs[i], rf_sphere_handles[i], gymapi.STATE_NONE)
        state['pose']['p'].fill((_rf_position_des[i, 0], _rf_position_des[i, 1], _rf_position_des[i, 2]))
        state['pose']['r'].fill((0, 0, 0, 1))
        state['vel']['linear'].fill((0, 0, 0))
        state['vel']['angular'].fill((0, 0, 0))
        gym.set_actor_rigid_body_states(envs[i], rf_sphere_handles[i], state, gymapi.STATE_ALL)
        #draw reaction force
        gym.draw_env_rigid_contacts(viewer, envs[i], gymapi.Vec3(255, 0.0, 0.0), rf_scale, False)
        verts[0, :] = _lf_position[i, :].view(-1)
        verts[1, :] = rf_scale*_Fr[i, :3].view(-1)
        verts[2, :] = _rf_position[i, :].view(-1)
        verts[3, :] = rf_scale*_Fr[i, 3:].view(-1)

        gym.add_lines(viewer, envs[i], 2, verts, colors)
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(_tau))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    gym.clear_lines(viewer)
    if log_data:
        logger['_lf_position'].append(_lf_position[0, :].view(-1).clone().numpy())
        logger['_rf_position'].append(_rf_position[0, :].view(-1).clone().numpy())
        logger['_lf_position_des'].append(_lf_position_des[0, :].view(-1).clone().numpy())
        logger['_rf_position_des'].append(_rf_position_des[0, :].view(-1).clone().numpy())
        logger['_swing_states'].append(_swing_states[0, :].view(-1).clone().numpy())
        logger['_Fr'].append(_Fr[0, :].view(-1).clone().numpy())
        logger['_com_position'].append(_com_position[0, :].view(-1).clone().numpy())
        logger['_com_vel'].append(_com_vel[0, :].view(-1).clone().numpy())
        logger['_phase'].append(_phase[0].clone().numpy())
        logger['_t_stamp'].append(step_count*sim_params.dt)
        step_count +=1
    # time.sleep(1.0)
if log_data:
    import pickle
    with open('log.pickle', 'wb') as handle:
        pickle.dump(logger, handle, protocol=pickle.HIGHEST_PROTOCOL)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

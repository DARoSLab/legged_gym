from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *


import math
import numpy as np
import torch
import random
import time

#pytorch3d https://github.com/facebookresearch/pytorch3d
from pytorch3d.transforms import quaternion_to_matrix, so3_log_map

# Cubic bezier interpolation between y0 and yf.  x is between 0 and 1
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
# def computeHeightSwingTrajectory(_p0, _p_mid, _p_f, phase, alpha, swing_time):
def computeHeightSwingTrajectory(_p0, _pf, phase, _swing_height, swing_time):
    _p = cubicBezier(_p0, _pf, phase)
    _pdot = cubicBezierFirstDerivative(_p0, _pf, phase) / swing_time
    _pddot = cubicBezierSecondDerivative(_p0, _pf, phase) / (swing_time*swing_time)

    _lift_idx = phase<0.5
    _p[_lift_idx, 2] = cubicBezier(_p0[_lift_idx, 2].view(-1, 1), _p0[_lift_idx, 2].view(-1, 1) + _swing_height, phase[_lift_idx] * 2).squeeze(-1)
    _pdot[_lift_idx, 2]= cubicBezierFirstDerivative(_p0[_lift_idx, 2].view(-1, 1), _p0[_lift_idx, 2].view(-1, 1) + _swing_height, phase[_lift_idx] * 2).squeeze(-1) * 2 / swing_time
    _pddot[_lift_idx, 2] = cubicBezierSecondDerivative(_p0[_lift_idx, 2].view(-1, 1), _p0[_lift_idx, 2].view(-1, 1) + _swing_height, phase[_lift_idx] * 2).squeeze(-1) * 4 / (swing_time * swing_time)

    _step_idx = torch.logical_not(_lift_idx)

    _p[_step_idx, 2] = cubicBezier(_p0[_step_idx, 2].view(-1, 1) + _swing_height, _pf[_step_idx, 2].view(-1, 1), phase[_step_idx] * 2 - 1).squeeze(-1)
    _pdot[_step_idx, 2] = cubicBezierFirstDerivative(_p0[_step_idx, 2].view(-1, 1) + _swing_height, _pf[_step_idx, 2].view(-1, 1), phase[_step_idx] * 2 - 1).squeeze(-1) * 2 / swing_time
    _pddot[_step_idx, 2] = cubicBezierSecondDerivative(_p0[_step_idx, 2].view(-1, 1) + _swing_height, _pf[_step_idx, 2].view(-1, 1), phase[_step_idx] * 2 - 1).squeeze(-1) * 4 / (swing_time * swing_time)

    return _p, _pdot, _pddot
def computeLiftSwingTrajectory(_p0, _p_mid, _pf, swing_phase, alpha=0.5, swing_time=0.33):

    _lift_idx = swing_phase<alpha
    _step_idx = torch.logical_not(_lift_idx)

    _p = torch.zeros_like(_p0)
    _pdot = torch.zeros_like(_p0)
    _pddot = torch.zeros_like(_p0)

    _p[_lift_idx, :] = cubicBezier(_p0[_lift_idx, :], _p_mid[_lift_idx, :], swing_phase[_lift_idx]/alpha)
    _pdot[_lift_idx, :] = cubicBezierFirstDerivative(_p0[_lift_idx, :], _p_mid[_lift_idx, :], swing_phase[_lift_idx]) / (alpha*swing_time)
    _pddot[_lift_idx, :] = cubicBezierSecondDerivative(_p0[_lift_idx, :], _p_mid[_lift_idx, :], swing_phase[_lift_idx]) / (alpha*alpha*swing_time*swing_time)

    _p[_step_idx, :] = cubicBezier(_p_mid[_step_idx, :], _pf[_step_idx, :], (swing_phase[_step_idx]-alpha)/(1.0-alpha))
    _pdot[_step_idx, :] = cubicBezierFirstDerivative(_p_mid[_step_idx, :], _pf[_step_idx, :], (swing_phase[_step_idx]-alpha)/(1.0-alpha)) / (alpha*swing_time)
    _pddot[_step_idx, :] = cubicBezierSecondDerivative(_p_mid[_step_idx, :], _pf[_step_idx, :], (swing_phase[_step_idx]-alpha)/(1.0-alpha)) / ((1-alpha)*(1-alpha)*swing_time*swing_time)

    return _p, _pdot, _pddot

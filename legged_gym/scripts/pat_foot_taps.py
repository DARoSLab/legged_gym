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

LEFT_FOOT = True
STEP_HEIGHT = 0.2
STEP_PERIOD = 3

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


np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# Add custom arguments
args = gymutil.parse_arguments(description="Pat RL foot tap test",
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
asset_options.fix_base_link = True 
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
pat_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
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

labd_idxs = []
rabd_idxs = []
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

    labd_idx = gym.find_actor_rigid_body_index(env, pat_handle, "L_hip", gymapi.DOMAIN_SIM)
    labd_idxs.append(labd_idx)
    rabd_idx = gym.find_actor_rigid_body_index(env, pat_handle, "R_hip", gymapi.DOMAIN_SIM)
    rabd_idxs.append(rabd_idx)

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


# Root body state tensor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor)
root_positions = root_tensor[:, 0:3]

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

while step_count < 1000 and not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
   
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    _lf_position = rb_states[lf_idxs, :3]
    _rf_position = rb_states[rf_idxs, :3]

    height = (STEP_HEIGHT/2)*math.sin(step_count/STEP_PERIOD)+(STEP_HEIGHT/2)

    for i in range(num_envs):
        if LEFT_FOOT:
            print(step_count, True, height)
            state = gym.get_actor_rigid_body_states(envs[i], lf_sphere_handles[i], gymapi.STATE_NONE)
            state['pose']['p'].fill((_lf_position_des[i, 0], _lf_position_des[i, 1], height))
            state['pose']['r'].fill((0, 0, 0, 1))
            state['vel']['linear'].fill((0, 0, 0))
            state['vel']['angular'].fill((0, 0, 0))
            gym.set_actor_rigid_body_states(envs[i], lf_sphere_handles[i], state, gymapi.STATE_ALL)
        else:
            state = gym.get_actor_rigid_body_states(envs[i], rf_sphere_handles[i], gymapi.STATE_NONE)
            state['pose']['p'].fill((_rf_position_des[i, 0], _rf_position_des[i, 1], height))
            state['pose']['r'].fill((0, 0, 0, 1))
            state['vel']['linear'].fill((0, 0, 0))
            state['vel']['angular'].fill((0, 0, 0))
            gym.set_actor_rigid_body_states(envs[i], rf_sphere_handles[i], state, gymapi.STATE_ALL)

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

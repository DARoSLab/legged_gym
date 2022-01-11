import matplotlib.pyplot as plt
import numpy as np
import pickle
from legged_gym.envs.pat.pat_config import PatCfg
_swing_height = PatCfg.foot_placement.swing_height
with open('log.pickle', 'rb') as handle:
    log = pickle.load(handle)

plt.plot(log['_t_stamp'], np.array(log['_rf_position'])[:, 2], label="rf")
plt.plot(log['_t_stamp'], np.array(log['_rf_position_des'])[:, 2], label="rf_des")
plt.plot(log['_t_stamp'], np.array(log['_lf_position'])[:, 2], label="lf")
plt.plot(log['_t_stamp'], np.array(log['_lf_position_des'])[:, 2], label="lf_des")
plt.plot(log['_t_stamp'], _swing_height*np.array(log['_swing_states'])[:, 0], label="left swing state")
plt.plot(log['_t_stamp'], _swing_height*np.array(log['_swing_states'])[:, 1], label="right swing state")
plt.plot(log['_t_stamp'], _swing_height*np.array(log['_phase']), label="phase")
plt.legend(loc="upper left")
plt.xlabel("t")
# print(np.array(log['_swing_states']))
plt.figure()
# plt.plot(log['_t_stamp'], np.array(log['_rf_position'])[:, 1], label="rf")
# plt.plot(log['_t_stamp'], np.array(log['_rf_position_des'])[:, 1], label="rf_des")
plt.plot(log['_t_stamp'], np.array(log['_lf_position'])[:, 1], label="lf")
plt.plot(log['_t_stamp'], np.array(log['_lf_position_des'])[:, 1], label="lf_des")
plt.plot(log['_t_stamp'], _swing_height*np.array(log['_swing_states'])[:, 0], label="left swing state")
# plt.plot(log['_t_stamp'], _swing_height*np.array(log['_swing_states'])[:, 1], label="right swing state")
plt.plot(log['_t_stamp'], _swing_height*np.array(log['_phase']), label="phase")
plt.legend(loc="upper left")
plt.xlabel("t")
plt.figure()
plt.plot(np.array(log['_com_position'])[:, 1], np.array(log['_com_vel'])[:, 1], 'x', label="_com_position")

plt.show()

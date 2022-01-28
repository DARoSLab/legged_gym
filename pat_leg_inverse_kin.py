import torch
import math


def pat_leg_inverse_kin(foot_pos):
    '''Takes cartesian foot position and returns joint angles for abd, hip and knee'''

    THIGH_LENGTH = 1
    SHANK_LENGTH = 1

    R = torch.tensor([[0., 0., -1.], 
                      [0., 1., 0.],
                      [1., 0., 0.]])

    cart_pos = torch.matmul(R, foot_pos)

    r = math.sqrt(cart_pos[0]**2 + cart_pos[1]**2 + cart_pos[2]**2)

    theta = torch.zeros(3)

    theta[0] = math.atan(cart_pos[1]/cart_pos[0])
    theta[1] = -math.acos((cart_pos[0]**2 + cart_pos[1]**2 + cart_pos[2]**2-(THIGH_LENGTH**2)-(SHANK_LENGTH**2))/(2*THIGH_LENGTH*SHANK_LENGTH))
    theta[2] = -math.asin(cart_pos[2]/r)-math.atan((SHANK_LENGTH*math.sin(theta[1]))/(THIGH_LENGTH+SHANK_LENGTH*math.cos(theta[1])))

    return theta

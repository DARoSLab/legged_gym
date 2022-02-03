import torch
import math


def pat_leg_inverse_kin(foot_pos):
    '''Takes cartesian foot position and returns joint angles for abd, hip and knee'''

    THIGH_LENGTH = 0.21
    SHANK_LENGTH = 0.195
    
    R = torch.tensor([[0., 0., -1.], 
                      [0., 1., 0.],
                      [1., 0., 0.]])

    cart_pos = torch.matmul(R, foot_pos)
    cart_pos = [cart_pos[0]-0.001, cart_pos[1]-0.045, cart_pos[2]-0.382]
    cart_pos = [0.0000001 if coord == 0 else coord for coord in cart_pos]

    theta = torch.zeros(3)
    print(cart_pos)
    theta[0] = math.atan(cart_pos[1]/cart_pos[0])
    theta[1] = -math.acos((cart_pos[0]**2 + cart_pos[1]**2 + cart_pos[2]**2-(THIGH_LENGTH**2)-(SHANK_LENGTH**2))/(2*THIGH_LENGTH*SHANK_LENGTH))
    theta[2] = -math.asin(cart_pos[2]/r)-math.atan((SHANK_LENGTH*math.sin(theta[1]))/(THIGH_LENGTH+SHANK_LENGTH*math.cos(theta[1])))

    return theta


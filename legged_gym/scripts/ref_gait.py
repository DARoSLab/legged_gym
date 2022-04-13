import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

def fk(th1, th2, l1=0.2, l2=0.2):
    rA = np.array([l1*np.cos(th1), l1*np.sin(th1)])
    rB = np.array([l1*np.cos(th1) + l2*np.cos(th1+th2),
                   l1*np.sin(th1) + l2*np.sin(th1+th2)])
    return rA, rB
def ik(x, y, l1=0.2, l2=0.2):
    q2 = np.arccos((x**2 + y**2 -l1**2-l2**2)/(2*l1*l2))
    q1 = np.arctan(y/(x+1e-8)) - np.arctan((l2*np.sin(q2))/(l1+l2*np.cos(q2)))
    if(q1>0):
        q1 = q1-np.pi
    return [q1, q2]
def ik3d(x, y, z, l2=0.2, l3=0.2):
    a = l3
    b = l2
    c = np.sqrt(x**2+y**2+z**2)

    q0 = np.arctan(y/(z+1e-8))
    q1 = np.arccos((b**2+c**2-a**2)/(2*b*c)) - np.arctan(x/np.sqrt(y**2 + z**2))
    q2 = np.pi - np.arccos((a**2 + b**2 - c**2)/(2*a*b))
    return [q0, q1, q2]
def swing_ref(phase, x_default=-0.001, y_default = -0.4, swing_height=0.25):
    y = 0
    if phase < np.pi:
        t = (1.0/np.pi)*phase
        y = y_default + swing_height*(-2*t**3+3*t**2)
    elif(phase < 2*np.pi):
        t = (1/np.pi)*phase -1.0
        y = y_default + swing_height*(2*t**3 - 3*t**2 + 1)
    else:
        pass
    [q1, q2] = ik(x_default, y)
    return x_default, y, q1, q2
def swing_ref3d(phase, x_default=-0.001, y_default=0.06, z_default = -0.35, swing_height=0.25):
    z = 0
    if phase < np.pi:
        t = (1.0/np.pi)*phase
        z = z_default + swing_height*(-2*t**3+3*t**2)
    elif(phase < 2*np.pi):
        t = (1/np.pi)*phase -1.0
        z = z_default + swing_height*(2*t**3 - 3*t**2 + 1)
    else:
        pass
    [q0, q1, q2] = ik3d(x_default, y_default, z)
    return x_default, y_default, z_default, q0, q1, q2
class DrawRobot:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        ani = animation.FuncAnimation(self.fig, self.animate, interval=2)
        self.th1 = np.linspace(-np.pi/2. - np.pi/4, np.pi, 100)
        self.th2 = np.linspace(np.pi/2., np.pi, 100)
        self.gait_period = 0.66
        plt.show()
    def animate(self, i):
        phase = 2*np.pi*(np.fmod(i*0.05, 0.66)/0.66)
        # th1 = -np.pi/2 -np.pi/4 + 0.08*np.sin(phase + np.pi/2)
        # th2 = np.pi/2.0 - 0.15*np.cos(phase)
        swing_height =  0.05
        [x, y, q1, q2] = swing_ref(phase, y_default = -0.35, swing_height=swing_height)
        [x3, y3, z3, q03, q13, q23] = swing_ref3d(phase, x_default=0.0, y_default=0.1,
                                            z_default = -0.35, swing_height=swing_height)
        [rA, rB] = fk(q1, q2)
        [rA3, rB3] = fk(-(q13)+ 3*np.pi/2, q23)
        # print("th1: {} q1: {} \n th2: {} q2: {}".format(th1, q1, th2, q2))
        circle1 = plt.Circle((rA[0], rA[1]), 0.02, color='r', zorder=2)
        circle2 = plt.Circle((rB[0], rB[1]), 0.01, color='g', zorder=2)
        circle3 = plt.Circle((0, 0), 0.02, color='r', zorder=2)
        circle13 = plt.Circle((rA3[0], rA3[1]), 0.02, color='b', zorder=2)
        circle23 = plt.Circle((rB3[0], rB3[1]), 0.01, color='g', zorder=2)
        circle33 = plt.Circle((0, 0), 0.02, color='r', zorder=2)

        self.ax.clear()
        self.ax.plot([0, rA[0]], [0, rA[1]], linewidth=8)
        self.ax.plot([rA[0], rB[0]], [rA[1], rB[1]], linewidth=4)
        self.ax.plot([0, rA3[0]], [0, rA3[1]], linewidth=8)
        self.ax.plot([rA3[0], rB3[0]], [rA3[1], rB3[1]], linewidth=4)
        self.ax.add_patch(circle1)
        self.ax.add_patch(circle2)
        self.ax.add_patch(circle3)
        self.ax.add_patch(circle13)
        self.ax.add_patch(circle23)
        self.ax.add_patch(circle33)

        plt.xlim([-0.4, 0.4])
        plt.ylim([-0.4, 0.4])
if __name__ =="__main__":
     DrawRobot()

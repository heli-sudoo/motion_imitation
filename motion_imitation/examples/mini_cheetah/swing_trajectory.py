#!/bin/usr/python3

import numpy as np

from utils import interpolation as interp

class FootSwingTrajectory():
    def __init__(self):
        self.ps = np.zeros(3)
        self.pf = np.zeros(3)
        self.h = 0.05
        self.p = np.zeros(3)
        self.v = np.zeros(3)

    def set_swing_height(self, h):
        self.h = h

    def set_start_location(self, ps):
        self.ps = ps

    def set_end_location(self, pf):
        self.pf = pf
    
    def get_position(self):
        return self.p
    
    def get_velocity(self):
        return self.v

    def computeTrajectory(self, phase, swingTime):
        """ Compute foot swing trajectory with a Bezier curver
        Input phase: where are we in the swing [0, 1]
              swingTime: time duration of a swing        
        """
        self.p = interp.CubicBezier(self.ps, self.pf, phase)
        self.v = interp.CubicBezierFirstDerivative(self.ps, self.pf, phase)/swingTime

        # use two Bezier curves for the vertial motion
        if phase < 0.5:
            zp = interp.CubicBezier(self.ps[2], self.pf[2] + self.h, phase*2)
            zv = interp.CubicBezierFirstDerivative(self.ps[2], self.pf[2]+self.h, phase*2)/(0.5*swingTime)
        else:
            zp = interp.CubicBezier(self.pf[2]+self.h, self.pf[2], phase*2 - 1)
            zv = interp.CubicBezierFirstDerivative(self.pf[2]+self.h, self.pf[2], phase*2 -1)/(0.5*swingTime)
        
        self.p[2] = zp
        self.v[2] = zv
__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import numpy as np
import matplotlib.pyplot as plt

class simple(object):
    def __init__(self, t, amp=1, freq=1, phase=0, bias=0):
        self.t = t
        self.amp = amp
        self.freq = freq
        self.phase = phase
        self.bias = bias

    def current_u(self, time):
        u = self.amp*np.cos(self.freq*time+self.phase)+self.bias
        return u

    def signal(self, t=None):
        """
        Calculate the full signal for the u function
        :param t: 1-D array
            The vector of time values
        :return s: 1-D array
            The vector of values constituting the signal s(t)
        """
        if t is None:
            t = self.t
        s = [self.current_u(time) for time in t]
        return s

class step(simple):
    def __init__(self, t, mag=1, *args, **kwargs):
        super(step, self).__init__(t, *args, **kwargs)
        self.half = self.t[len(t) / 2]
        self.mag = mag

    def current_u(self, time):
        if time <=self.half:
            u = np.cos(time)
        elif time > self.half:
            u = np.cos(time)+self.mag
        return u

class transient(simple):
    def __init__(self, t, mag=1, *args, **kwargs):
        super(transient, self).__init__(t, *args, **kwargs)
        self.trans_max = self.t[len(t) / 2+100]
        self.trans_min = self.t[len(t) / 2-100]
        self.mag = mag

    def current_u(self, time):
        if time <= self.trans_min or time > self.trans_max:
            u = np.cos(time)
        elif time > self.trans_min or time <= self.trans_max:
            u = np.cos(time)+self.mag
        return u

if __name__ == "__main__":
    t = np.linspace(0,100,1000)
    x = simple(t)
    y = simple(t, freq=2)
    z = transient(t,2)
    plt.plot(t, x.signal(), t, y.signal(), t, z.signal())
    plt.show()
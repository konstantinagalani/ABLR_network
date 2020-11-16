
import numpy as np
import math

class Branin_function():


    def call(self,inpt):
        """Branin test function
        2-d input
        The number of variables n = 2.
        constraints:
        -5 <= x <= 10, 0 <= y <= 15
        three global optima:  (-pi, 12.275), (pi, 2.275), (9.42478, 2.475),
        where branin = 0.397887"""
        
        res = []
        if len(inpt.shape) == 1: inpt = inpt.reshape((1,inpt.shape[0]))
        for data in inpt:
            x = data[0]
            y = data[1]
            result = (y-(5.1/(4*math.pi**2))*x**2+5*x/math.pi-6)**2
            result += 10*(1-1/(8*math.pi))*math.cos(x)+10
            res.append(result)
        return np.array(res).astype(np.float32)

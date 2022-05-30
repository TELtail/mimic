import numpy as np
import math

def count_nan(data):
    """
    in: data
    out: num
    """
    num = np.isnan(data)
    num = np.all()
    return num


a = np.arange(0,12)
a = a.reshape(3,-1)
print(a)
print(count_nan(a))
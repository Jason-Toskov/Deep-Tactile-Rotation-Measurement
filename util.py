import numpy as np

def dist_to_guess(p_base, guess):
    return np.sqrt((p_base.x - guess[0])**2 + (p_base.y - guess[1])**2 + (p_base.z - guess[2])**2)

def vector3ToNumpy(v):
    return np.array([v.x, v.y, v.z])
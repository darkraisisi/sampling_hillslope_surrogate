import numpy as np

def step():
    # Define the physical parameters
    r, c, i, d, s = 2.1, 2.9, -0.7, 0.04, 0.4 
    Wo, a, Et, Eo, k, b, C = 5e-4, 4.02359478109, 0.021, 0.084, 0.05, 0.28, 1e-4
    alpha = np.log(Wo/C)/a

def step(B,D,g):
    dB_dt_step = (1-(1-i)*np.exp(-1*D/d))*(r*B*(1-B/c))-g*B/(s+B)
    dD_dt_step = Wo*np.exp(-1*a*D)-np.exp(-1*B/b)*(Et+np.exp(-1*D/k)*(Eo-Et))-C
    return dB_dt_step, dD_dt_step